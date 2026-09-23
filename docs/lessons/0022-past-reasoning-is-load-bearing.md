Past reasoning in the Duck's history is load-bearing: stripping it to fit more requests halved the score (exp-035).

The Flash-Next template renders every past turn's reasoning, which is 35% of the prompt. Dropping it (with older
turns' instructions and images) and lowering the history window to 22,528 tokens halved the prompt and raised the
requests running from 3.1 to 4.2, but the model then re-derived its state at every call: replies grew 38%, the share of
turns that acted fell from 0.64 to 0.39, preemptions went from 21 to 615, and the public-25 score fell from 7.86 to 3.58
(below 36 of 37 base runs). Throughput counts only if the calls keep their quality; measure acting share and reply
length next to requests per minute before trusting a serving-side gain. The base's 32,768-token window and uncapped
output stay until an arm beats them.
