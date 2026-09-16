The ARC-AGI-3-Agents framework stores ACTION6 coordinates on the shared GameAction enum member, so with one thread per game (Swarm) clicks can be sent with another game's coordinates.

- `Agent.do_action_request` reads `action.action_data` after `choose_action` returns; every game thread calls
  `GameAction.ACTION6.set_data(...)` on the same singleton. The window is tiny but there are thousands of clicks
  across 25 concurrent games.
- Fix in our adapter: `agent/my_agent.py` overrides `do_action_request` to send the per-game data kept on the
  `Driver`. Test: `tests/test_submission.py::test_click_coordinates_do_not_go_through_the_shared_enum`.
- Same caution for `reasoning`: never rely on attributes of the shared enum member.
