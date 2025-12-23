import torch
from torch.optim.optimizer import Optimizer
import typing


class SimpleLBFGS(Optimizer):
    state: typing.Dict[str, typing.Any]
    def __init__(self, params, lr=1.0, history_size=20 , max_iter=20):
        defaults = dict(lr=lr, history_size=history_size , max_iter=max_iter)
        super().__init__(params, defaults)

        self.state['n_iter'] = 0
        self.state['old_params'] = None
        self.state['old_grad'] = None
        self.state['s_history'] = []
        self.state['y_history'] = []

    def _gather_flat_grad(self):
        return torch.cat([
            p.grad.view(-1)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ])

    def _gather_flat_params(self):
        return torch.cat([
            p.data.view(-1)
            for group in self.param_groups
            for p in group['params']
        ])

    def _set_flat_params(self, flat_params):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(flat_params[offset:offset + numel].view_as(p))
                offset += numel

    def _lbfgs_step_once(self, closure):
        if closure is None:
            raise RuntimeError("LBFGS requires a closure")

        loss = closure()
        grad = self._gather_flat_grad()
        params = self._gather_flat_params()

        state = self.state
        lr = self.param_groups[0]['lr']
        m = self.param_groups[0]['history_size']

        with torch.no_grad():
            if state['n_iter'] == 0:
                direction = -grad
            else:
                direction = self._two_loop_recursion(grad)

                direction = -direction

            new_params = params + lr * direction
            self._set_flat_params(new_params)
        
        with torch.enable_grad():
            loss = closure()
        
        with torch.no_grad():
            new_grad = self._gather_flat_grad()

            s = new_params - params
            y = new_grad - grad

            if torch.dot(s, y) > 1e-10:
                state['s_history'].append(s)
                state['y_history'].append(y)

                if len(state['s_history']) > m:
                    state['s_history'].pop(0)
                    state['y_history'].pop(0)

            state['n_iter'] += 1

        return loss

    def step(self, closure):
        max_iter = self.param_groups[0]['max_iter']
        for _ in range(max_iter):
            loss = self._lbfgs_step_once(closure)
        return loss

    def _two_loop_recursion(self, grad):
        s_list = self.state['s_history']
        y_list = self.state['y_history']

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
