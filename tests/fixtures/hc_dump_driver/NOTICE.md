# Test fixture: two shortened harness request logs (tests/test_hc_dump_driver.py)

`ls20-9607627b_p0_requests.jsonl` (113,613 bytes, sha256 a56c268a...) and `sb26-7fbdac44_p0_requests.jsonl` (66,984
bytes, sha256 cd6e33ca...) are samples of the `<game>_p0_requests.jsonl` logs that Daniel Franzen's harness
(Apache-2.0) wrote in our CPU bed run for exp-078 (scripts/franzen_bed.py: his real harness on the public games,
answered by the bed's scripted mock model, so the "reasoning" is mock text, not a model's). Source files, from the
session scratchpad's `bed-exp078/`: `ls20-9607627b_p0_requests.jsonl` sha256 e585fd80... and
`sb26-7fbdac44_p0_requests.jsonl` sha256 bbd9aa7c....

How they were cut (a scratch script, parsing JSON only):
- the first 14 (ls20) and 10 (sb26) requests, each with the response record that followed it;
- every string longer than 72 characters (message text, reasoning, tool output, each tool-call argument, the tool
  descriptions) cut to its first 72 characters plus ` [...N chars, sha256 xxxxxxxx]`, so equal texts stay equal and
  different texts stay different;
- every image data URL replaced by one 1x1 PNG.

Checked when they were made: which request extends the next one is unchanged (ls20: not at requests 5 and 8; sb26:
not at 5), and so is the number of distinct assistant turns (13 and 8; ls20 has two without a tool call). The line
format (`request` / `response` records, `_arc3_control` keys, usage) is untouched. Do not edit them.
