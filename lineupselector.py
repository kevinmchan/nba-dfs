import pulp as plp


class FanDuelOptimizer:
    def __init__(self, target):
        self._target = target

    def add_top_n_lineups(self, players, n=10):
        candidates = players.copy()
        best_players = []
        for i in range(n):
            candidates[f"selection_{i + 1}"] = self.add_lineup_selection(candidates)
            best_players.append(candidates.loc[lambda x: x[self._target] == (
                        x[f"selection_{i + 1}"] * x[self._target]).max(), "player_id"].iloc[0])
            players = players.merge(candidates[["player_id", f"selection_{i + 1}"]], how="left")
            candidates = players.query("player_id not in @best_players")

        return players

    def add_lineup_selection(self, players):
        """
        Integer program roster optimization

        decision_variable: x_i for each player i
        maximize: sum(x_i * fpts_i)
        subject to:
            budget: sum(x_i * salary_i) <= 60_000
            roster: for each role r, sum(x_i * role_i_r) <= slots_r
            num_players: sum(x_i) = 9
            no_blocklist_players
            binary_decision_var: for each player i, x_i in (0, 1)
        """
        # model
        opt_model = plp.LpProblem(name="Lineup_Optimization")

        # decision variables
        x_vars = [
            plp.LpVariable(cat=plp.LpBinary, name=f"start_player_{i}")
            for i in players.player_id
        ]

        # budget
        opt_model.addConstraint(
            plp.LpConstraint(
                e=plp.lpSum(salary * player for (salary, player) in zip(players["salary"], x_vars)),
                sense=plp.LpConstraintLE,
                rhs=60_000,
                name="salary_cap"
            )
        )

        # blocklist players
        opt_model.addConstraint(
            plp.LpConstraint(
                e=plp.lpSum(blocklisted * player for (blocklisted, player) in zip(players["blocklisted"], x_vars)),
                sense=plp.LpConstraintEQ,
                rhs=0,
                name="player_blocklist"
            )
        )

        # max players
        opt_model.addConstraint(
            plp.LpConstraint(
                e=plp.lpSum(player for player in x_vars),
                sense=plp.LpConstraintLE,
                rhs=9,
                name="max_roster"
            )
        )

        # max position players
        max_positions = {
            "PG": 2,
            "SG": 2,
            "SF": 2,
            "PF": 2,
            "C": 1
        }
        for pos, max_players in max_positions.items():
            opt_model.addConstraint(
                plp.LpConstraint(
                    e=plp.lpSum(player * position for (player, position) in zip(x_vars, players[f"position_{pos}"])),
                    sense=plp.LpConstraintLE,
                    rhs=max_players,
                    name=f"max_positions_{pos}"
                )
            )

        objective = plp.lpSum(fpts * player for (fpts, player) in zip(players[self._target], x_vars))
        opt_model.sense = plp.LpMaximize
        opt_model.setObjective(objective)
        opt_model.solve()
        return [x.varValue for x in x_vars]