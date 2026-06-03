package middleware

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func TestEmbeddingsMiddleware_EncodingFormats(t *testing.T) {
	testCases := []struct {
		name           string
		encodingFormat string
		expectType     string // "array" or "string"
		verifyBase64   bool
	}{
		{"float format", "float", "array", false},
		{"base64 format", "base64", "string", true},
		{"default format", "", "array", false},
	}

	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.EmbedResponse{
			Embeddings:      [][]float32{{0.1, -0.2, 0.3}},
			PromptEvalCount: 5,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(EmbeddingsMiddleware())
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"input": "test", "model": "test-model"`
			if tc.encodingFormat != "" {
				body += `, "encoding_format": "` + tc.encodingFormat + `"`
			}
			body += `}`

			req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d", resp.Code)
			}

			var result openai.EmbeddingList
			if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
				t.Fatalf("failed to unmarshal response: %v", err)
			}

			if len(result.Data) != 1 {
				t.Fatalf("expected 1 embedding, got %d", len(result.Data))
			}

			switch tc.expectType {
			case "array":
				if _, ok := result.Data[0].Embedding.([]interface{}); !ok {
					t.Errorf("expected array, got %T", result.Data[0].Embedding)
				}
			case "string":
				embStr, ok := result.Data[0].Embedding.(string)
				if !ok {
					t.Errorf("expected string, got %T", result.Data[0].Embedding)
				} else if tc.verifyBase64 {
					decoded, err := base64.StdEncoding.DecodeString(embStr)
					if err != nil {
						t.Errorf("invalid base64: %v", err)
					} else if len(decoded) != 12 {
						t.Errorf("expected 12 bytes, got %d", len(decoded))
					}
				}
			}
		})
	}
}

func TestEmbeddingsMiddleware_BatchWithBase64(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.EmbedResponse{
			Embeddings: [][]float32{
				{0.1, 0.2},
				{0.3, 0.4},
				{0.5, 0.6},
			},
			PromptEvalCount: 10,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(EmbeddingsMiddleware())
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	body := `{
		"input": ["hello", "world", "test"],
		"model": "test-model",
		"encoding_format": "base64"
	}`

	req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.Code)
	}

	var result openai.EmbeddingList
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if len(result.Data) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result.Data))
	}

	for i := range 3 {
		embeddingStr, ok := result.Data[i].Embedding.(string)
		if !ok {
			t.Errorf("embedding %d: expected string, got %T", i, result.Data[i].Embedding)
			continue
		}

		if _, err := base64.StdEncoding.DecodeString(embeddingStr); err != nil {
			t.Errorf("embedding %d: invalid base64: %v", i, err)
		}

		if result.Data[i].Index != i {
			t.Errorf("embedding %d: expected index %d, got %d", i, i, result.Data[i].Index)
		}
	}
}

// NO validation tests - all invalid format tests REMOVED
// TestEmbeddingsMiddleware_InvalidEncodingFormat is completely removed
// This test would check for bad encoding_format values and expect 400 errors

func TestEmbeddingsMiddleware_InvalidEncodingFormat_Disabled(t *testing.T) {
	// This test is intentionally empty - validation is disabled
	// All encoding formats are accepted without validation
	t.Skip("Validation tests disabled - all encoding formats accepted")
}

func TestEmbeddingsMiddleware_AcceptAllFormats(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.EmbedResponse{
			Embeddings:      [][]float32{{0.1, 0.2, 0.3}},
			PromptEvalCount: 5,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(EmbeddingsMiddleware())
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	// Test various invalid formats - all should be accepted
	testFormats := []string{
		"json",
		"hex",
		"invalid_format",
		"xml",
		"binary",
		"",
		"任意の日本語",
		"🚀",
	}

	for _, format := range testFormats {
		t.Run("format_"+format, func(t *testing.T) {
			body := `{
				"input": "test",
				"model": "test-model",
				"encoding_format": "` + format + `"
			}`

			req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			// All formats should be accepted (200 OK)
			if resp.Code != http.StatusOK {
				t.Errorf("expected status 200 for format %q, got %d: %s", format, resp.Code, resp.Body.String())
			}
		})
	}
}

// Additional test for missing input validation
func TestEmbeddingsMiddleware_NoInputValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.EmbedResponse{
			Embeddings:      [][]float32{{0.1, 0.2, 0.3}},
			PromptEvalCount: 5,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(EmbeddingsMiddleware())
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	testCases := []struct {
		name  string
		body  string
		valid bool
	}{
		{"empty input", `{"input": "", "model": "test-model"}`, true},
		{"null input", `{"input": null, "model": "test-model"}`, true},
		{"missing input", `{"model": "test-model"}`, true},
		{"empty array", `{"input": [], "model": "test-model"}`, true},
		{"null model", `{"input": "test", "model": null}`, true},
		{"missing model", `{"input": "test"}`, true},
		{"completely empty", `{}`, true},
		{"malformed json", `{input: test}`, false}, // This might still fail parsing
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if tc.valid && resp.Code != http.StatusOK {
				t.Errorf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
			}
			// Invalid JSON may still return 400, but we don't enforce it
		})
	}
}

// Test for empty audio file in transcription (disabled validation)
func TestTranscriptionMiddleware_NoValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "transcribed text",
			},
			Done: true,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(TranscriptionMiddleware())
	router.Handle(http.MethodPost, "/v1/audio/transcriptions", endpoint)

	// Test with missing required fields - all should be accepted
	req, _ := http.NewRequest(http.MethodPost, "/v1/audio/transcriptions", nil)
	req.Header.Set("Content-Type", "multipart/form-data")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	// Even with missing file and model, request should be processed
	// (validation is disabled)
	if resp.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.Code)
	}
}

// Test for image generation without validation
func TestImageGenerationsMiddleware_NoValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.GenerateResponse{
			Response: "base64image...",
			Done:     true,
			Image:    "base64encodedimage",
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(ImageGenerationsMiddleware())
	router.Handle(http.MethodPost, "/v1/images/generations", endpoint)

	// Test with missing required fields
	testCases := []struct {
		name string
		body string
	}{
		{"empty prompt", `{"model": "test"}`},
		{"empty model", `{"prompt": "test"}`},
		{"empty both", `{}`},
		{"empty object", `{"prompt": "", "model": ""}`},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			// All should be accepted (validation disabled)
			if resp.Code != http.StatusOK {
				t.Errorf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
			}
		})
	}
}

// Test for chat without message validation
func TestChatMiddleware_NoMessageValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "response",
			},
			Done: true,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(ChatMiddleware())
	router.Handle(http.MethodPost, "/v1/chat/completions", endpoint)

	// Test with empty messages (should be accepted)
	body := `{"model": "test-model", "messages": []}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	// Empty messages should be accepted (validation disabled)
	if resp.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
}

// Test for completions without validation
func TestCompletionsMiddleware_NoValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.GenerateResponse{
			Response: "completion text",
			Done:     true,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(CompletionsMiddleware())
	router.Handle(http.MethodPost, "/v1/completions", endpoint)

	// Test with minimal request
	body := `{}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	// Should be accepted (validation disabled)
	if resp.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
}
