package middleware

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func testPropsMap(m map[string]api.ToolProperty) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

var argsComparer = cmp.Comparer(func(a, b api.ToolCallFunctionArguments) bool {
	return cmp.Equal(a.ToMap(), b.ToMap())
})

var propsComparer = cmp.Comparer(func(a, b *api.ToolPropertiesMap) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return cmp.Equal(a.ToMap(), b.ToMap())
})

const (
	prefix = `data:image/jpeg;base64,`
	image  = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=`
)

var (
	False = false
	True  = true
)

func captureRequestMiddleware(capturedRequest any) gin.HandlerFunc {
	return func(c *gin.Context) {
		bodyBytes, _ := io.ReadAll(c.Request.Body)
		c.Request.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		json.Unmarshal(bodyBytes, capturedRequest)
		c.Next()
	}
}

func sseDataFrames(body string) []string {
	frames := strings.Split(body, "\n\n")
	data := make([]string, 0, len(frames))
	for _, frame := range frames {
		frame = strings.TrimSpace(frame)
		if !strings.HasPrefix(frame, "data: ") {
			continue
		}
		data = append(data, strings.TrimPrefix(frame, "data: "))
	}
	return data
}

func TestChatWriter_StreamMixedThinkingAndContentEmitsSplitChunks(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(recorder)

	writer := &ChatWriter{
		stream:        true,
		streamOptions: &openai.StreamOptions{IncludeUsage: true},
		id:            "chatcmpl-test",
		BaseWriter:    BaseWriter{ResponseWriter: context.Writer},
	}

	response := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "reasoning",
			Content:  "final answer",
		},
		Done:       true,
		DoneReason: "stop",
		Metrics: api.Metrics{
			PromptEvalCount: 3,
			EvalCount:       2,
		},
	}

	data, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}

	if _, err = writer.Write(data); err != nil {
		t.Fatalf("write response: %v", err)
	}

	frames := sseDataFrames(recorder.Body.String())
	if len(frames) != 4 {
		t.Fatalf("expected 4 SSE data frames, got %d", len(frames))
	}
	if frames[3] != "[DONE]" {
		t.Fatalf("expected final frame [DONE], got %q", frames[3])
	}
}

func TestChatWriter_StreamSingleChunkPathStillEmitsOneChunk(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(recorder)

	writer := &ChatWriter{
		stream:     true,
		id:         "chatcmpl-test",
		BaseWriter: BaseWriter{ResponseWriter: context.Writer},
	}

	response := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Content: "single chunk",
		},
		Done:       true,
		DoneReason: "stop",
	}

	data, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}

	if _, err = writer.Write(data); err != nil {
		t.Fatalf("write response: %v", err)
	}

	frames := sseDataFrames(recorder.Body.String())
	if len(frames) != 2 {
		t.Fatalf("expected 2 SSE data frames, got %d", len(frames))
	}
	if frames[1] != "[DONE]" {
		t.Fatalf("expected final frame [DONE], got %q", frames[1])
	}
}

func TestChatMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.ChatRequest
	}

	var capturedRequest *api.ChatRequest

	testCases := []testCase{
		{
			name: "chat handler",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "chat handler with streaming usage",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Hello"}
				],
				"stream": true,
				"stream_options": {"include_usage": true}
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &True,
			},
		},
		{
			name: "chat handler with image content",
			body: `{
				"model": "test-model",
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": "Hello"
							},
							{
								"type": "image_url",
								"image_url": {
									"url": "` + prefix + image + `"
								}
							}
						]
					}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello",
					},
					{
						Role: "user",
						Images: []api.ImageData{
							func() []byte {
								img, _ := base64.StdEncoding.DecodeString(image)
								return img
							}(),
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "chat handler with tools",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather?"},
					{"role": "assistant", "tool_calls": [{"id": "id", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}}]}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather?",
					},
					{
						Role: "assistant",
						ToolCalls: []api.ToolCall{
							{
								ID: "id",
								Function: api.ToolCallFunction{
									Name:      "get_weather",
									Arguments: testArgs(map[string]any{"location": "Paris"}),
								},
							},
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(ChatMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/chat", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			defer func() { capturedRequest = nil }()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
			}

			if diff := cmp.Diff(&tc.req, capturedRequest, argsComparer, propsComparer); diff != "" {
				t.Fatalf("requests did not match: %+v", diff)
			}
		})
	}
}

func TestCompletionsMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.GenerateRequest
	}

	var capturedRequest *api.GenerateRequest

	testCases := []testCase{
		{
			name: "completions handler",
			body: `{
				"model": "test-model",
				"prompt": "Hello"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "Hello",
				Options: map[string]any{
					"frequency_penalty": 0.0,
					"presence_penalty":  0.0,
					"temperature":       1.0,
					"top_p":             1.0,
				},
				Stream: &False,
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(CompletionsMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/generate", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d", resp.Code)
			}

			capturedRequest = nil
		})
	}
}

func TestEmbeddingsMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.EmbedRequest
	}

	var capturedRequest *api.EmbedRequest

	testCases := []testCase{
		{
			name: "embed handler single input",
			body: `{
				"input": "Hello",
				"model": "test-model"
			}`,
			req: api.EmbedRequest{
				Input: "Hello",
				Model: "test-model",
			},
		},
		{
			name: "embed handler batch input",
			body: `{
				"input": ["Hello", "World"],
				"model": "test-model"
			}`,
			req: api.EmbedRequest{
				Input: []any{"Hello", "World"},
				Model: "test-model",
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(EmbeddingsMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d", resp.Code)
			}

			capturedRequest = nil
		})
	}
}

func TestListMiddleware(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(ListMiddleware())
	router.Handle(http.MethodGet, "/api/tags", func(c *gin.Context) {
		c.JSON(http.StatusOK, api.ListResponse{
			Models: []api.ListModelResponse{
				{
					Name:       "test-model",
					ModifiedAt: time.Unix(int64(1686935002), 0).UTC(),
				},
			},
		})
	})

	req, _ := http.NewRequest(http.MethodGet, "/api/tags", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.Code)
	}
}

func TestRetrieveMiddleware(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(RetrieveMiddleware())
	router.Handle(http.MethodGet, "/api/show/:model", func(c *gin.Context) {
		c.JSON(http.StatusOK, api.ShowResponse{
			ModifiedAt: time.Unix(int64(1686935002), 0).UTC(),
		})
	})

	req, _ := http.NewRequest(http.MethodGet, "/api/show/test-model", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.Code)
	}
}

func TestImageGenerationsMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.GenerateRequest
	}

	var capturedRequest *api.GenerateRequest

	testCases := []testCase{
		{
			name: "image generation basic",
			body: `{
				"model": "test-model",
				"prompt": "a beautiful sunset"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "a beautiful sunset",
			},
		},
		{
			name: "image generation with size",
			body: `{
				"model": "test-model",
				"prompt": "a beautiful sunset",
				"size": "512x768"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "a beautiful sunset",
				Width:  512,
				Height: 768,
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(ImageGenerationsMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/generate", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			defer func() { capturedRequest = nil }()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
			}
		})
	}
}

func TestImageEditsMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.GenerateRequest
	}

	var capturedRequest *api.GenerateRequest

	testImage := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	decodedImage, _ := base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")

	testCases := []testCase{
		{
			name: "image edit basic",
			body: `{
				"model": "test-model",
				"prompt": "make it blue",
				"image": "` + testImage + `"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "make it blue",
				Images: []api.ImageData{decodedImage},
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(ImageEditsMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/generate", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			defer func() { capturedRequest = nil }()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
			}
		})
	}
}

func zstdCompress(t *testing.T, data []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	w, err := zstd.NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestResponsesMiddlewareZstd(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		useZstd   bool
		wantCode  int
		wantModel string
	}{
		{
			name:     "plain JSON",
			body:     `{"model": "test-model", "input": "Hello"}`,
			wantCode: http.StatusOK,
		},
		{
			name:     "zstd compressed",
			body:     `{"model": "test-model", "input": "Hello"}`,
			useZstd:  true,
			wantCode: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedRequest *api.ChatRequest

			gin.SetMode(gin.TestMode)
			router := gin.New()
			router.Use(ResponsesMiddleware(), captureRequestMiddleware(&capturedRequest))
			router.Handle(http.MethodPost, "/v1/responses", func(c *gin.Context) {
				c.Status(http.StatusOK)
			})

			var bodyReader io.Reader
			if tt.useZstd {
				bodyReader = bytes.NewReader(zstdCompress(t, []byte(tt.body)))
			} else {
				bodyReader = strings.NewReader(tt.body)
			}

			req, _ := http.NewRequest(http.MethodPost, "/v1/responses", bodyReader)
			req.Header.Set("Content-Type", "application/json")
			if tt.useZstd {
				req.Header.Set("Content-Encoding", "zstd")
			}

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != tt.wantCode {
				t.Fatalf("expected status %d, got %d", tt.wantCode, resp.Code)
			}
		})
	}
}
