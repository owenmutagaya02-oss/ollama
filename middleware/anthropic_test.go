package middleware

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
)

func captureAnthropicRequest(capturedRequest any) gin.HandlerFunc {
	return func(c *gin.Context) {
		bodyBytes, _ := io.ReadAll(c.Request.Body)
		c.Request.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		_ = json.Unmarshal(bodyBytes, capturedRequest)
		c.Next()
	}
}

func testProps(m map[string]api.ToolProperty) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

func TestAnthropicMessagesMiddleware_Basic(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.ChatRequest
	}

	var capturedRequest *api.ChatRequest
	stream := true

	testCases := []testCase{
		{
			name: "basic message",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &False,
			},
		},
		{
			name: "with system prompt",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"system": "You are helpful.",
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "system", Content: "You are helpful."},
					{Role: "user", Content: "Hello"},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &False,
			},
		},
		{
			name: "with tools",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "What's the weather?"}
				],
				"tools": [{
					"name": "get_weather",
					"description": "Get current weather",
					"input_schema": {
						"type": "object",
						"properties": {
							"location": {"type": "string"}
						},
						"required": ["location"]
					}
				}]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "What's the weather?"},
				},
				Tools: []api.Tool{
					{
						Type: "function",
						Function: api.ToolFunction{
							Name:        "get_weather",
							Description: "Get current weather",
							Parameters: api.ToolFunctionParameters{
								Type:     "object",
								Required: []string{"location"},
								Properties: testProps(map[string]api.ToolProperty{
									"location": {Type: api.PropertyType{"string"}},
								}),
							},
						},
					},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &False,
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(AnthropicMessagesMiddleware(), captureAnthropicRequest(&capturedRequest))
	router.Handle(http.MethodPost, "/v1/messages", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			defer func() { capturedRequest = nil }()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("unexpected status code: %d, body: %s", resp.Code, resp.Body.String())
			}

			if capturedRequest == nil {
				t.Fatal("request was not captured")
			}

			if capturedRequest.Model != tc.req.Model {
				t.Errorf("model mismatch: got %q, want %q", capturedRequest.Model, tc.req.Model)
			}

			if diff := cmp.Diff(tc.req.Messages, capturedRequest.Messages,
				cmpopts.IgnoreUnexported(api.ToolCallFunctionArguments{}, api.ToolPropertiesMap{})); diff != "" {
				t.Errorf("messages mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAnthropicWriter_NonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "Hello there!",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics: api.Metrics{
				PromptEvalCount: 10,
				EvalCount:       5,
			},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{"model": "test-model", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.Code)
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if result.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", result.Role)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Text == nil || *result.Content[0].Text != "Hello there!" {
		t.Errorf("expected text 'Hello there!', got %v", result.Content[0].Text)
	}
	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("expected input_tokens 10, got %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 5 {
		t.Errorf("expected output_tokens 5, got %d", result.Usage.OutputTokens)
	}
}

func TestHasWebSearchTool(t *testing.T) {
	tests := []struct {
		name     string
		tools    []anthropic.Tool
		expected bool
	}{
		{
			name:     "no tools",
			tools:    nil,
			expected: false,
		},
		{
			name: "regular tool only",
			tools: []anthropic.Tool{
				{Type: "custom", Name: "get_weather"},
			},
			expected: false,
		},
		{
			name: "web search tool",
			tools: []anthropic.Tool{
				{Type: "web_search_20250305", Name: "web_search"},
			},
			expected: true,
		},
		{
			name: "mixed tools",
			tools: []anthropic.Tool{
				{Type: "custom", Name: "get_weather"},
				{Type: "web_search_20250305", Name: "web_search"},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasWebSearchTool(tt.tools)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestExtractQueryFromToolCall(t *testing.T) {
	tests := []struct {
		name     string
		tc       *api.ToolCall
		expected string
	}{
		{
			name: "valid query",
			tc: &api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      "web_search",
					Arguments: makeArgs("query", "test search"),
				},
			},
			expected: "test search",
		},
		{
			name: "empty arguments",
			tc: &api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "web_search",
				},
			},
			expected: "",
		},
		{
			name: "no query key",
			tc: &api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      "web_search",
					Arguments: makeArgs("other", "value"),
				},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractQueryFromToolCall(tt.tc)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func makeArgs(key string, value any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	args.Set(key, value)
	return args
}

func TestWebSearchServerToolUseID(t *testing.T) {
	tests := []struct {
		msgID    string
		expected string
	}{
		{"msg_abc123", "srvtoolu_abc123"},
		{"msg_", "srvtoolu_"},
		{"nomsgprefix", "srvtoolu_nomsgprefix"},
	}
	for _, tt := range tests {
		got := serverToolUseID(tt.msgID)
		if got != tt.expected {
			t.Errorf("serverToolUseID(%q) = %q, want %q", tt.msgID, got, tt.expected)
		}
	}
}

func TestWebSearchNoWebSearchTool(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "Normal response",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 5},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{"model":"test-model","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if len(result.Content) != 1 || result.Content[0].Type != "text" {
		t.Fatalf("expected single text block, got %d blocks", len(result.Content))
	}
	if *result.Content[0].Text != "Normal response" {
		t.Errorf("expected text 'Normal response', got %q", *result.Content[0].Text)
	}
}

func TestWebSearchToolPresent_ModelDoesNotCallIt_NonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "I can answer that without searching.",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 12, EvalCount: 8},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"What is 2+2?"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if len(result.Content) != 1 || result.Content[0].Type != "text" {
		t.Fatalf("expected single text block, got %+v", result.Content)
	}
	if *result.Content[0].Text != "I can answer that without searching." {
		t.Errorf("unexpected text: %q", *result.Content[0].Text)
	}
	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
}

func TestWebSearchToolPresent_ModelDoesNotCallIt_Streaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		chunks := []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "Hello "},
				Done:    false,
			},
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "world"},
				Done:    false,
			},
			{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant", Content: ""},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 5},
			},
		}
		c.Writer.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			_, _ = c.Writer.Write(data)
		}
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"stream":true,
		"messages":[{"role":"user","content":"Hi"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	events := parseSSEEvents(t, resp.Body.String())

	if len(events) == 0 {
		t.Fatal("expected SSE events, got none")
	}

	if events[0].event != "message_start" {
		t.Errorf("first event should be message_start, got %q", events[0].event)
	}
}

func TestWebSearchStreamResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	text := "Here is the answer."

	response := anthropic.MessagesResponse{
		ID:    "msg_test123",
		Type:  "message",
		Role:  "assistant",
		Model: "test-model",
		Content: []anthropic.ContentBlock{
			{
				Type:  "server_tool_use",
				ID:    "srvtoolu_test123",
				Name:  "web_search",
				Input: queryArgs("test query"),
			},
			{
				Type:      "web_search_tool_result",
				ToolUseID: "srvtoolu_test123",
				Content: []anthropic.WebSearchResult{
					{Type: "web_search_result", URL: "https://example.com", Title: "Example"},
				},
			},
			{
				Type: "text",
				Text: &text,
			},
		},
		StopReason: "end_turn",
		Usage:      anthropic.Usage{InputTokens: 20, OutputTokens: 10},
	}

	rec := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(rec)

	innerWriter := &AnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		stream:     true,
		id:         "msg_test123",
	}
	wsWriter := &WebSearchAnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		inner:      innerWriter,
		stream:     true,
		req:        anthropic.MessagesRequest{Model: "test-model"},
	}

	if err := wsWriter.streamResponse(response); err != nil {
		t.Fatalf("streamResponse error: %v", err)
	}

	events := parseSSEEvents(t, rec.Body.String())

	if len(events) == 0 {
		t.Fatal("expected events, got none")
	}
}

func TestWebSearchStreamingImmediateTakeover(t *testing.T) {
	gin.SetMode(gin.TestMode)

	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.ChatResponse{
			Model:      "test-model",
			Message:    api.Message{Role: "assistant", Content: "After search."},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 20, EvalCount: 10},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Result", URL: "https://example.com", Content: "content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		chunks := []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "Preface "},
				Done:    false,
			},
			{
				Model: "test-model",
				Message: api.Message{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_ws_stream_1",
							Function: api.ToolCallFunction{
								Name:      "web_search",
								Arguments: makeArgs("query", "latest updates"),
							},
						},
					},
				},
				Done: false,
			},
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "ignored chunk"},
				Done:    false,
			},
			{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant"},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 9, EvalCount: 4},
			},
		}
		c.Writer.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			_, _ = c.Writer.Write(data)
		}
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"stream":true,
		"messages":[{"role":"user","content":"Find updates"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	events := parseSSEEvents(t, resp.Body.String())
	if countEventsByName(events, "message_start") != 1 {
		t.Fatalf("expected exactly one message_start, got %d", countEventsByName(events, "message_start"))
	}
	if countEventsByName(events, "message_stop") != 1 {
		t.Fatalf("expected exactly one message_stop, got %d", countEventsByName(events, "message_stop"))
	}

	textDeltas := collectTextDeltas(t, events)
	if !containsString(textDeltas, "Preface ") {
		t.Fatalf("expected passthrough text delta, got %v", textDeltas)
	}
	if !containsString(textDeltas, "After search.") {
		t.Fatalf("expected post-search text delta, got %v", textDeltas)
	}
	if containsString(textDeltas, "ignored chunk") {
		t.Fatalf("unexpected text from chunks after takeover: %v", textDeltas)
	}
}

// Helper functions
type sseEvent struct {
	event string
	data  string
}

func parseSSEEvents(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	var currentEvent string
	var currentData strings.Builder

	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			currentData.WriteString(strings.TrimPrefix(line, "data: "))
		} else if line == "" && currentEvent != "" {
			events = append(events, sseEvent{event: currentEvent, data: currentData.String()})
			currentEvent = ""
			currentData.Reset()
		}
	}
	return events
}

func eventNames(events []sseEvent) []string {
	names := make([]string, len(events))
	for i, e := range events {
		names[i] = e.event
	}
	return names
}

func countEventsByName(events []sseEvent, eventName string) int {
	count := 0
	for _, event := range events {
		if event.event == eventName {
			count++
		}
	}
	return count
}

func collectTextDeltas(t *testing.T, events []sseEvent) []string {
	t.Helper()

	var deltas []string
	for _, event := range events {
		if event.event != "content_block_delta" {
			continue
		}

		var delta anthropic.ContentBlockDeltaEvent
		if err := json.Unmarshal([]byte(event.data), &delta); err != nil {
			t.Fatalf("failed to unmarshal content_block_delta: %v", err)
		}
		if delta.Delta.Type == "text_delta" {
			deltas = append(deltas, delta.Delta.Text)
		}
	}

	return deltas
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

// Dummy functions for compilation
var False = false

func enableCloudForTest(t *testing.T) {
	// NO-OP - cloud always allowed
}

func queryArgs(query string) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	args.Set("query", query)
	return args
}
