import torch
from torch.optim.optimizer import Optimizer
import typing


class SimpleLBFGS(Optimizer):
    state: typing.Dict[str, typing.Any]

    def __init__(
        self,
        params,
        lr=1.0,
        history_size=20,
        max_iter=10,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    ):
        defaults = dict(
            lr=lr,
            history_size=history_size,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
        )
        super().__init__(params, defaults)

        self.state["n_iter"] = 0
        self.state["old_params"] = None
        self.state["old_grad"] = None
        self.state["s_history"] = []
        self.state["y_history"] = []

    def _gather_flat_grad(self):
        return torch.cat(
            [
                p.grad.view(-1)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
        )

    def _add_update(self, step_size, update):
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.data.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
                offset += numel

    def step(self, closure):
        if closure is None:
            raise RuntimeError("LBFGS requires a closure")

        loss = closure()

        with torch.no_grad():
            grad = self._gather_flat_grad()

            state = self.state
            lr = self.param_groups[0]["lr"]
            m = self.param_groups[0]["history_size"]
            if state["n_iter"] == 0:
                direction = -grad
            else:
                direction = self._two_loop_recursion(grad)

                direction = -direction

                self._add_update(lr, direction)

        loss = closure()
        prev_loss = loss

        for _ in range(self.param_groups[0]["max_iter"]):
            with torch.no_grad():
                if self.state["n_iter"] == 0:
                    direction = -grad
                else:
                    direction = -self._two_loop_recursion(grad)

                self._add_update(lr, direction)

            loss = closure()
            new_grad = self._gather_flat_grad()

            if new_grad.norm() < self.param_groups[0]["tolerance_grad"]:
                break

            if (
                abs(loss.detach() - prev_loss)
                < self.param_groups[0]["tolerance_change"]
            ):
                break

            # update history
            s = lr * direction
            y = new_grad - grad

            if torch.dot(s, y) > 1e-10:
                self.state["s_history"].append(s)
                self.state["y_history"].append(y)

                if len(self.state["s_history"]) > m:
                    self.state["s_history"].pop(0)
                    self.state["y_history"].pop(0)

            grad = new_grad
            prev_loss = loss
            self.state["n_iter"] += 1

        with torch.no_grad():
            new_grad = self._gather_flat_grad()

            s = direction * self.param_groups[0]["lr"]
            y = new_grad - grad

            if torch.dot(s, y) > 1e-10:
                state["s_history"].append(s)
                state["y_history"].append(y)

                if len(state["s_history"]) > m:
                    state["s_history"].pop(0)
                    state["y_history"].pop(0)

            state["n_iter"] += 1

        return loss

    def _two_loop_recursion(self, grad):
        s_list = self.state["s_history"]
        y_list = self.state["y_history"]

        q = grad.clone()
        alpha = []

        # backward process
        for s, y in reversed(list(zip(s_list, y_list))):
            rho = 1.0 / torch.dot(y, s)
            a = rho * torch.dot(s, q)
            alpha.append(a)
            q = q - a * y

        if len(s_list) > 0:
            s, y = s_list[-1], y_list[-1]
            gamma = torch.dot(s, y) / torch.dot(y, y)
        else:
            gamma = 1.0

        r = gamma * q

        # forward process
        for s, y, a in zip(s_list, y_list, reversed(alpha)):
            rho = 1.0 / torch.dot(y, s)
            beta = rho * torch.dot(y, r)
            r = r + s * (a - beta)

        return r
