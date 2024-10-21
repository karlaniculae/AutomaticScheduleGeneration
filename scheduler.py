class Scheduler:
    def __init__(self,days,intervals,materials,professors,rooms,scheduler_state=None):
        """initialize the scheduler"""
        self.days = days
        self.intervals = intervals
        self.materials = materials
        self.professors = professors
        self.rooms = rooms
        self.professor_constraints = {prof: self.professors[prof]['Constrangeri'] for prof in self.professors}
        self.professor_subjects = {prof: self.professors[prof]['Materii'] for prof in self.professors}
        self.room_capacity = {room: self.rooms[room]['Capacitate'] for room in self.rooms}
        self.room_subjects = {room: self.rooms[room]['Materii'] for room in self.rooms}
        self.preferred_days, self.preferred_intervals = self.constrains(self.professor_constraints)
        self.plain_intervals=[(intv['start'], intv['end']) for intv in self.intervals]
        #calculate professor assignments or create initial schedule depending of the data
        if scheduler_state is not None:
            self.scheduler_state = scheduler_state
            self.professor_assignments=self.nr_of_ass(self.scheduler_state)
        else:
            self.scheduler_state, self.professor_assignments = self.create_initial_schedule()
           
        


    
    def parse_interval(self,interval):
        """
        transform intervals like 10-14 into intervals of 2 hours
        """
        if '-' in interval:
            start, end = map(int, interval.split('-'))
            return [f"{hour}-{hour + 2}" for hour in range(start, end, 2)]
        return [interval]
    
    def constrains(self,professor_constraints):
        """split the constrains of the professors in intervals and days"""
        preferred_days = {}
        preferred_intervals = {}

        for prof, constraints in professor_constraints.items():
            preferred_days[prof] = [
                day for day in constraints
                if day in {'Luni', 'Marti', 'Miercuri', 'Joi', 'Vineri'}
            ]
            all_intervals = [
                interval for interval in constraints
                if '-' in interval and not interval.startswith('!')
            ]
            
            normalized_intervals = []
            for interval in all_intervals:
                normalized_intervals.extend(self.parse_interval(interval))
            
            preferred_intervals[prof] = normalized_intervals
        return preferred_days,  preferred_intervals
    def create_initial_schedule(self):
        """create the initial schedule which respects all the hard constrains""" 
        room_counts = {subject: sum(subject in subjects for subjects in self.room_subjects.values()) for subject in self.materials.keys()}
        schedule = {
            day: {interval: {room: None for room in self.rooms} for interval in self.plain_intervals}
            for day in self.days
        }

        professor_assignments = {prof: {'assignments': 0, 'constraints': self.professor_constraints[prof], 'subjects': self.professor_subjects[prof]} for prof in self.professors}
        professor_sessions = set() 
        sorted_rooms = sorted(self.rooms, key=lambda room: self.room_capacity[room], reverse=True)

        sorted_subjects = sorted(self.materials.items(), key=lambda item: room_counts[item[0]])
        
        #start with the subject with the most students
        for subject, total_student_count in sorted_subjects:
            student_count = total_student_count  
            sorted_rooms_copy = sorted_rooms.copy()
            while student_count > 0:
                best_room = None
                min_waste = float('inf')

                for room in sorted_rooms_copy:
                    if subject in self.room_subjects[room]:
                        waste = abs(self.room_capacity[room] - student_count) 
                        #find the room with the minimum waste
                        if waste < min_waste:
                            min_waste = waste
                            best_room = room
                if best_room is not None:
                    sorted_rooms_copy.remove(best_room)
                    for day in self.days:
                    
                        for interval in self.plain_intervals:
                            
                            if student_count <= 0:
                                break
                            for prof in self.professors:
                            
                                if subject in self.professor_subjects[prof] and professor_assignments[prof]['assignments'] < 7:
                                    session_key = (day, interval, prof)
                                    if session_key not in professor_sessions and schedule[day][interval][best_room] is None:
                                        #add the session to the schedule
                                        schedule[day][interval][best_room] = (prof, subject)
                                        professor_sessions.add((day, interval, prof))
                                        assigned_students = min(self.room_capacity[best_room], student_count)
                                        student_count -= assigned_students
                                        professor_assignments[prof]['assignments'] += 1
        
        schedule_final={day: schedule[day] for day in self.days}

        return schedule_final, professor_assignments
    def nr_of_ass(self, scheduler_state):
        """calculate how many times a professor teaches in that week"""
        professor_assignments = {prof: {'assignments': 0, 'subjects': self.professor_subjects[prof]} for prof in self.professors}
        for day in scheduler_state:
            for interval in scheduler_state[day]:
                for room in scheduler_state[day][interval]:
                    session = scheduler_state[day][interval][room]
                    if session:
                        prof, subj = session
                        professor_assignments[prof]['assignments'] += 1
        #sort them in ascending order by the number of assignments
        sorted_professor_assignments = sorted(professor_assignments.items(), key=lambda x: x[1]['assignments'])
        return {prof: details for prof, details in sorted_professor_assignments}

        

        