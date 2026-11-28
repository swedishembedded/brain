<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 621. An agent's next call prefilled its previous answer again, because the prefix cache kept prompt blocks only

**Lesson:** once a chat history replays each answer as its sampled ids (knowledge 620), call k+1's prompt is call k's
prompt, its answer and the turns after it, exactly. The engines indexed a sequence's blocks only when its prompt was
prefilled, and let the decoded rows go with the sequence. So call k+1 found call k's prompt in the cache and prefilled
the answer again. On the tiny Qwen3 engine behind the OpenAI route, the second call of a two-call session was served
64 of 111 prompt ids from cache (the first prompt's full blocks: 67 ids, 8-row blocks).

**Fix:** the scheduler releases a finished sequence with the ids its rows hold: its prompt and the answer it was fed
back, never the last sampled id, which no step read (`PagedDecoder::release_finished`). A decoder with a prefix cache
keeps what those rows computed. The paged-KV engine (qwen3) indexes the full blocks in the sequence's namespace, taking
the cache's own reference, evictable like any cached block. The sequence's references go back as before, so nothing
is charged twice. Under the exact-decode regime it keeps the prompt's blocks only, since decode rows are computed at
another row count than a cold prefill computes them. The ring engine (fixed-size state, the snapshot cache) files the
finished slot's state beside the prompt's entry (`Entry::beside`, `take_row_beside`). Other samples of the same prompt
still resume from the prompt; the next turn of the answer's conversation replaces it. The entry has no hidden row, so
an exact repeat of it is prefilled. The same call now serves 72 of 111 ids (every full block of the prompt and the
answer: (67 + 6 - 1) / 8 * 8); a ring session's next call resumes after 27 of 36 ids (prompt 24, answer 4, last id
unread). Spec: `catalog` `it::replayed_sessions` (both routes), `qwen35`
`it::edgerock_serve::the_next_call_of_a_replayed_answer_resumes_after_the_answer` (warm equals cold),
`model` `snapshot_prefix::tests::an_answer_filed_beside_its_prompt_keeps_the_prompt`.

The looped `VisitPool` engines (inkling2) and ricursive snapshot the sequence's state row at release the same way
(`VisitPool::file_answer`; ricursive's `Filed::Answer`); a beside entry is never handed out for an exact repeat
(`Entry::is_beside`). The served 27B's engine (`Qwen35GgufEngine`) files it in its paged layout only, and not under
exact decode, where a resume inside a 64-token recurrent chunk rounds otherwise than a cold prefill.
