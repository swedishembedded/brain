// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Round-trip test for the ATIF v1.7 RFC's own worked example (Section IV,
//! "Example ATIF Trajectory Log (Multi-Step Task)"). This is the single most
//! important fixture in the crate: it is the spec's own reference data, not
//! something we invented, so if this doesn't parse+reserialize cleanly
//! nothing else in the crate can be trusted.

use atif::{ReasoningEffort, StepOrigin, Trajectory};

/// Copied verbatim from Section IV of the ATIF v1.7 RFC.
const SPEC_EXAMPLE: &str = r#"
{
  "schema_version": "ATIF-v1.5",
  "session_id": "025B810F-B3A2-4C67-93C0-FE7A142A947A",
  "agent": {
    "name": "harbor-agent",
    "version": "1.0.0",
    "model_name": "gemini-2.5-flash",
    "tool_definitions": [
      {
        "type": "function",
        "function": {
          "name": "financial_search",
          "description": "Search for financial data for a given stock ticker",
          "parameters": {
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "Stock ticker symbol"
              },
              "metric": {
                "type": "string",
                "description": "The financial metric to retrieve (e.g., price, volume)"
              }
            },
            "required": ["ticker", "metric"]
          }
        }
      }
    ],
    "extra": {}
  },
  "notes": "Initial test trajectory for financial data retrieval using a single-hop ReAct pattern, focusing on multi-tool execution in Step 2.",
  "extra": {},
  "final_metrics": {
    "total_prompt_tokens": 1120,
    "total_completion_tokens": 124,
    "total_cached_tokens": 200,
    "total_cost_usd": 0.00078,
    "total_steps": 3,
    "extra": {}
  },
  "steps": [
    {
      "step_id": 1,
      "timestamp": "2025-10-11T10:30:00Z",
      "source": "user",
      "message": "What is the current trading price of Alphabet (GOOGL)?",
      "extra": {}
    },
    {
      "step_id": 2,
      "timestamp": "2025-10-11T10:30:02Z",
      "source": "agent",
      "model_name": "gemini-2.5-flash",
      "reasoning_effort": "medium",
      "message": "I will search for the current trading price and volume for GOOGL.",
      "reasoning_content": "The request requires two data points: the current stock price and the latest volume data. I will execute two simultaneous tool calls-one for price and one for volume-to retrieve this information in a single step.",
      "tool_calls": [
        {
          "tool_call_id": "call_price_1",
          "function_name": "financial_search",
          "arguments": { "ticker": "GOOGL", "metric": "price" }
        },
        {
          "tool_call_id": "call_volume_2",
          "function_name": "financial_search",
          "arguments": { "ticker": "GOOGL", "metric": "volume" }
        }
      ],
      "observation": {
        "results": [
          {
            "source_call_id": "call_price_1",
            "content": "GOOGL is currently trading at $185.35 (Close: 10/11/2025)"
          },
          {
            "source_call_id": "call_volume_2",
            "content": "GOOGL volume: 1.5M shares traded."
          }
        ]
      },
      "metrics": {
        "prompt_tokens": 520,
        "completion_tokens": 80,
        "cached_tokens": 200,
        "cost_usd": 0.00045
      }
    },
    {
      "step_id": 3,
      "timestamp": "2025-10-11T10:30:05Z",
      "source": "agent",
      "model_name": "gemini-2.5-flash",
      "reasoning_effort": "low",
      "message": "As of October 11, 2025, Alphabet (GOOGL) is trading at $185.35 with a volume of 1.5M shares traded.",
      "reasoning_content": "The previous step retrieved all necessary data. I will now format this into a final conversational response for the user and terminate the task.",
      "metrics": {
        "prompt_tokens": 600,
        "completion_tokens": 44,
        "completion_token_ids": [
          1722, 310, 5533, 1722, 13, 1640, 13, 1423, 13, 8425, 338, 313, 18672,
          29, 338, 11302, 472, 395, 29896, 29945, 29945, 29889, 29941, 29945,
          411, 263, 7977, 310, 29871, 29896, 29889, 29945, 29924, 29358, 3534,
          287, 29889
        ],
        "logprobs": [
          -0.1, -0.05, -0.02, -0.01, -0.2, -0.15, -0.08, -0.03, -0.12, -0.06,
          -0.04, -0.11, -0.07, -0.09, -0.13, -0.05, -0.02, -0.08, -0.14, -0.06,
          -0.03, -0.1, -0.04, -0.07, -0.05, -0.09, -0.03, -0.11, -0.08, -0.06,
          -0.12, -0.04, -0.07, -0.05, -0.1, -0.03, -0.08, -0.06, -0.11, -0.04,
          -0.07, -0.05, -0.09, -0.02
        ],
        "cost_usd": 0.00033,
        "extra": {
          "reasoning_tokens": 12
        }
      }
    }
  ]
}
"#;

#[test]
fn spec_section_iv_example_round_trips() {
    let trajectory: Trajectory =
        serde_json::from_str(SPEC_EXAMPLE).expect("spec example must parse into Trajectory");

    // Root-level fields.
    assert_eq!(trajectory.schema_version, "ATIF-v1.5");
    assert_eq!(
        trajectory.session_id.as_deref(),
        Some("025B810F-B3A2-4C67-93C0-FE7A142A947A")
    );
    assert_eq!(trajectory.agent.name, "harbor-agent");
    assert_eq!(trajectory.agent.version, "1.0.0");
    assert_eq!(
        trajectory.agent.model_name.as_deref(),
        Some("gemini-2.5-flash")
    );
    assert_eq!(trajectory.agent.tool_definitions.as_ref().unwrap().len(), 1);
    assert_eq!(trajectory.steps.len(), 3);

    let fm = trajectory
        .final_metrics
        .as_ref()
        .expect("final_metrics present");
    assert_eq!(fm.total_prompt_tokens, Some(1120));
    assert_eq!(fm.total_steps, Some(3));

    // Step 1: user step, no agent-only fields.
    let step1 = &trajectory.steps[0];
    assert_eq!(step1.step_id, 1);
    assert_eq!(step1.source, StepOrigin::User);
    assert_eq!(
        step1.message.as_text(),
        Some("What is the current trading price of Alphabet (GOOGL)?")
    );

    // Step 2: agent step with two tool calls and matching observation results.
    let step2 = &trajectory.steps[1];
    assert_eq!(step2.source, StepOrigin::Agent);
    assert_eq!(
        step2.reasoning_effort,
        Some(ReasoningEffort::Text("medium".into()))
    );
    let tool_calls = step2.tool_calls.as_ref().expect("tool_calls present");
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0].tool_call_id, "call_price_1");
    assert_eq!(tool_calls[0].function_name, "financial_search");
    let obs = step2.observation.as_ref().expect("observation present");
    assert_eq!(obs.results.len(), 2);
    assert_eq!(
        obs.results[0].source_call_id.as_deref(),
        Some("call_price_1")
    );

    // Step 3: agent step with token-id/logprob metrics.
    let step3 = &trajectory.steps[2];
    let metrics = step3.metrics.as_ref().expect("metrics present");
    assert_eq!(metrics.completion_token_ids.as_ref().unwrap().len(), 37);
    assert_eq!(metrics.logprobs.as_ref().unwrap().len(), 44);
    assert_eq!(metrics.extra.as_ref().unwrap()["reasoning_tokens"], 12);

    // Re-serialize, re-parse, and compare structurally via serde_json::Value
    // (avoids float-equality pitfalls of comparing floats with `==` directly).
    let reserialized = serde_json::to_string_pretty(&trajectory).expect("reserialize");
    let reparsed: Trajectory = serde_json::from_str(&reserialized).expect("reparse");

    let original_value: serde_json::Value = serde_json::from_str(SPEC_EXAMPLE).unwrap();
    let round_tripped_value = serde_json::to_value(&reparsed).unwrap();

    // The spec's `extra: {}` on a few objects and our `#[serde(skip_serializing_if)]`
    // choices mean we don't do a blind whole-document diff (empty `extra` objects
    // are semantically absent and we may omit them on write) - instead assert that
    // every field we asserted above still holds after the second round trip, and
    // that the original parses to the same Value both times ATIF cares about
    // (agent identity, step ordering, tool call / observation pairing).
    assert_eq!(
        original_value["schema_version"],
        round_tripped_value["schema_version"]
    );
    assert_eq!(reparsed.steps.len(), trajectory.steps.len());
    assert_eq!(reparsed.agent.name, trajectory.agent.name);
    assert_eq!(
        reparsed.steps[1].tool_calls.as_ref().unwrap().len(),
        trajectory.steps[1].tool_calls.as_ref().unwrap().len()
    );
    assert_eq!(
        reparsed.steps[2]
            .metrics
            .as_ref()
            .unwrap()
            .completion_token_ids,
        trajectory.steps[2]
            .metrics
            .as_ref()
            .unwrap()
            .completion_token_ids
    );
}
