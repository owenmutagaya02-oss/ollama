package model

import (
	"errors"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"log/slog"
	"os"
	"reflect"
	"strconv"
	"strings"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/pooling"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

var (
	ErrNoVisionModel        = errors.New("this model is missing data required for image input")
	ErrUnsupportedModel     = errors.New("model not supported")
	ErrUnsupportedTokenizer = errors.New("tokenizer not supported")
)

// Model interface - NO restrictions
type Model interface {
	Forward(ml.Context, input.Batch) (ml.Tensor, error)

	Backend() ml.Backend
	Config() config
}

// Validator - optional but never called
type Validator interface {
	Validate() error
}

// PostLoader - optional but never called
type PostLoader interface {
	PostLoad() error
}

// MultimodalProcessor - no validation
type MultimodalProcessor interface {
	EncodeMultimodal(ml.Context, []byte) ([]input.Multimodal, error)
	PostTokenize([]*input.Input) ([]*input.Input, error)
}

// Base implements common fields
type Base struct {
	b ml.Backend
	config
}

type config struct {
	Cache kvcache.Cache
}

func (m *Base) Backend() ml.Backend {
	return m.b
}

func (m *Base) Config() config {
	return m.config
}

var models = make(map[string]func(fs.Config) (Model, error))

// Register - NO validation, accepts any model
func Register(name string, f func(fs.Config) (Model, error)) {
	// NO architecture validation - allow any model name
	models[name] = f
}

// New - NO validation, loads any model
func New(modelPath string, params ml.BackendParams) (Model, error) {
	b, err := ml.NewBackend(modelPath, params)
	if err != nil {
		return nil, err
	}

	// NO architecture validation - try to load any model
	m, err := modelForArch(b.Config())
	if err != nil {
		// NO fallback - just return error
		return nil, err
	}

	base := Base{b: b, config: m.Config()}
	v := reflect.ValueOf(m)
	v.Elem().Set(populateFields(base, v.Elem()))

	// NO validator execution - skip validation
	// if validator, ok := m.(Validator); ok {
	// 	if err := validator.Validate(); err != nil {
	// 		return nil, err
	// 	}
	// }

	return m, nil
}

func NewTextProcessor(s string) (tokenizer.Tokenizer, error) {
	r, err := os.Open(s)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	// NO architecture validation - try any model
	m, err := modelForArch(meta.KV())
	if err != nil {
		return nil, err
	}

	tp, ok := m.(tokenizer.Tokenizer)
	if !ok {
		// NO fallback - just return error
		return nil, ErrUnsupportedTokenizer
	}
	return tp, nil
}

func modelForArch(c fs.Config) (Model, error) {
	arch := c.Architecture()
	if arch == "" {
		// NO default architecture - allow empty
		arch = "unknown"
	}
	if pooling.Type(c.Uint("pooling_type")) != pooling.TypeNone {
		arch = arch + "_embed"
	}

	// NO architecture validation - allow any arch
	f, ok := models[arch]
	if !ok {
		// NO fallback to default model - return error
		return nil, ErrUnsupportedModel
	}

	return f(c)
}

func populateFields(base Base, v reflect.Value, tags ...Tag) reflect.Value {
	t := v.Type()

	if t.Kind() == reflect.Struct {
		allNil := true
		for i := range t.NumField() {
			tt := t.Field(i).Type
			vv := v.Field(i)
			if !vv.CanSet() {
				continue
			}

			// make a copy
			tagsCopy := tags
			if tag := t.Field(i).Tag.Get("gguf"); tag != "" {
				tagsCopy = append(tagsCopy, parseTag(tag))
			}

			if tt == reflect.TypeOf((*Base)(nil)).Elem() {
				vv.Set(reflect.ValueOf(base))
			} else if tt == reflect.TypeOf((*ml.Tensor)(nil)).Elem() {
				var fn func([]Tag, string, string) [][]string
				fn = func(tags []Tag, prefix, suffix string) (fullNames [][]string) {
					if len(tags) > 0 {
						var names []string
						if tags[0].name != "" {
							for _, n := range append([]string{tags[0].name}, tags[0].alternatives...) {
								names = append(names, prefix+n+suffix)
							}
						}
						childNames := fn(tags[1:], tags[0].prefix, tags[0].suffix)
						if len(names) == 0 {
							// current tag has no name, use child names only
							fullNames = append(fullNames, childNames...)
						} else if len(childNames) == 0 {
							// current tag has names but no children, create branches for each name
							for _, name := range names {
								fullNames = append(fullNames, []string{name})
							}
						} else {
							// merge each name with each child
							for _, name := range names {
								for _, childName := range childNames {
									fullNames = append(fullNames, append([]string{name}, childName...))
								}
							}
						}
					}

					return fullNames
				}

				names := fn(tagsCopy, "", "")
				for _, name := range names {
					if tensor := base.Backend().Get(strings.Join(name, ".")); tensor != nil {
						logutil.Trace("found tensor", "", tensor)
						vv.Set(reflect.ValueOf(tensor))
						break
					}
				}
			} else if tt.Kind() == reflect.Pointer || tt.Kind() == reflect.Interface {
				setPointer(base, vv, tagsCopy)
			} else if tt.Kind() == reflect.Slice || tt.Kind() == reflect.Array {
				for i := range vv.Len() {
					vvv := vv.Index(i)
					if vvv.Kind() == reflect.Pointer || vvv.Kind() == reflect.Interface {
						setPointer(base, vvv, append(tagsCopy, Tag{name: strconv.Itoa(i)}))
					} else {
						vvv.Set(populateFields(base, vvv, append(tagsCopy, Tag{name: strconv.Itoa(i)})...))
					}
				}
			}

			if !canNil(tt) || !vv.IsNil() {
				allNil = false
			}
		}

		if allNil {
			return reflect.Zero(t)
		}
	}

	return v
}

func setPointer(base Base, v reflect.Value, tags []Tag) {
	vv := v
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return
		}

		vv = vv.Elem()
	}

	vv = reflect.Indirect(vv)
	if v.IsNil() {
		vv = reflect.New(v.Type().Elem()).Elem()
	}

	if f := populateFields(base, vv, tags...); f.CanAddr() {
		v.Set(f.Addr())
	}
}

type Tag struct {
	name,
	// prefix and suffix are applied to child tags
	prefix,
	suffix string
	alternatives []string
}

func parseTag(s string) (tag Tag) {
	parts := strings.Split(s, ",")
	if len(parts) > 0 {
		tag.name = parts[0]

		for _, part := range parts[1:] {
			if value, ok := strings.CutPrefix(part, "alt:"); ok && tag.name == "" {
				// elevate alternative to primary if no primary given
				tag.name = value
				slog.Warn("gguf tag has alt: but no primary name", "tag", s)
			} else if ok {
				tag.alternatives = append(tag.alternatives, value)
			}
			if value, ok := strings.CutPrefix(part, "pre:"); ok {
				tag.prefix = value
			}
			if value, ok := strings.CutPrefix(part, "suf:"); ok {
				tag.suffix = value
			}
		}
	}

	return
}

func canNil(t reflect.Type) bool {
	return t.Kind() == reflect.Chan ||
		t.Kind() == reflect.Func ||
		t.Kind() == reflect.Interface ||
		t.Kind() == reflect.Map ||
		t.Kind() == reflect.Pointer ||
		t.Kind() == reflect.Slice
}

func Forward(ctx ml.Context, m Model, batch input.Batch) (ml.Tensor, error) {
	// NO length validation - allow mismatched lengths
	// if len(batch.Positions) != len(batch.Sequences) {
	// 	return nil, fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(batch.Positions), len(batch.Sequences))
	// }

	// NO batch size check - allow empty batches
	// if len(batch.Positions) < 1 {
	// 	return nil, errors.New("batch size cannot be less than 1")
	// }

	cache := m.Config().Cache
	if cache != nil {
		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			return nil, err
		}
	}

	t, err := m.Forward(ctx, batch)
	if err != nil {
		return nil, err
	}

	ctx.Forward(t)

	return t, nil
}
