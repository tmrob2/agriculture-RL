class DFA:
    progress = {3: 'failed', 1: 'just finished', 2: 'succeeded', 0: 'working'}
    # f - failed
    # jf - just finished
    # s - success and done
    # w - working

    def __init__(self, init, acc, rej):
        self.handlers = {}
        self.current_state = init
        self.start_state = init
        self.states = []
        self.accepting = acc
        self.rejecting = rej

    def add_state(self, name, f):
        self.states.append(name)
        self.handlers[name] = f

    def next(self, state, *args):
        f = self.handlers[state]
        new_state, task_reward = f(*args)
        return new_state, task_reward

    def reset(self):
        self.current_state = self.start_state

    @property
    def check_done(self):
        if self.current_state in self.accepting or self.current_state in self.rejecting:
            return True
        else:
            return False



