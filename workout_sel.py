class workout_sel():
    def __init__(self):
        self.state = 'idle'
        self.workouts = []
        self.work_cnt = 0


    def load_workout(self, ollama_response: dict):
        """
        ollama_response = {excercise : '', priority : '', reason : ''}
        """
        recommend_workout = ollama_response['recommendations']
        self.workouts = sorted(recommend_workout, key=lambda x: x['priority'])
        self.state = 'work' if self.workouts else 'idle'
        self.work_cnt = 0

    def current_workout(self):
        """
        send current workout data
        if not, send none
        """
        if self.state == 'work' and self.work_cnt < len(self.workouts):
            return self.workouts[self.work_cnt]
        else:
            return None

    def next_workout(self, work_done: bool):
        """
        work_done is from workout_check module. send next exercise data after receive work_done signal
        """
        if self.state == 'work' and work_done:
            self.work_cnt += 1
            if self.work_cnt >= len(self.workouts):
                self.state = 'idle'
                return None
        return self.current_workout()