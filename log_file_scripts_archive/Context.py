from datetime import datetime

class Context():
    
    def __init__(self, filename):
        self.activities = []
        self.events = []
        self.filename = filename
    
    def print_context(self):
        f = open('data/context/{}_context.txt'.format(self.filename[:2]), 'w+')
        
        f.write('Filename: {}\n'.format(self.filename))
        for ii in range(len(self.activities)):
            if 'feedback' in self.activities[ii].keys():
                if 'backbutton' in self.activities[ii].keys():
                    activity_line = '{} Name: {}, Feedback: {}, Backbutton: {}\n'.format(
                            self.activities[ii]['start'].strftime("%H:%M:%S.%f"), 
                            self.activities[ii]['name'], 
                            self.activities[ii]['feedback'], 'True')
                else:
                    activity_line = '{} Name: {}, Feedback: {}, Backbutton: {}\n'.format(
                            self.activities[ii]['start'].strftime("%H:%M:%S.%f"), 
                            self.activities[ii]['name'], 
                            self.activities[ii]['feedback'], 'False')
            else:
                if 'backbutton' in self.activities[ii].keys():
                    activity_line = '{} Name: {}, Backbutton: {}\n'.format(
                            self.activities[ii]['start'].strftime("%H:%M:%S.%f"), 
                            self.activities[ii]['name'],'True')
                else:
                    activity_line = '{} Name: {}, Backbutton: {}\n'.format(
                            self.activities[ii]['start'].strftime("%H:%M:%S.%f"), 
                            self.activities[ii]['name'], 'False')
            f.write(activity_line)
            
            
            for jj in range(len(self.events[ii])):
                if len(self.events[ii][jj]['action']) > 0:
                    action_line = '\t {}, Event: {}, Action: {}\n'.format(
                            self.events[ii][jj]['time'].strftime("%H:%M:%S.%f"),
                            self.events[ii][jj]['event'],
                            self.events[ii][jj]['action'])
                else:
                    action_line = '\t {}, Event: {}\n'.format(
                            self.events[ii][jj]['time'].strftime("%H:%M:%S.%f"),
                            self.events[ii][jj]['event'])
                f.write(action_line)
            f.write('\n______________________\n')
            
        f.close()
        print('Wrote to file:', 'data/context/' + self.filename[:2]+'_context.txt')
        
    def get_entry(self, keyword, line):
        
        ind = line.find(keyword +'"')
        
        if ind == -1:
            return ''
        
        entry = line[ind+len(keyword)+3:]
        
        ind2 = entry.find('"')
        entry = entry[:ind2]
        
        return entry
    
    def get_timestamp(self, line):
        
        ind = line.find('"time"')
        timestamp = int(line[ind+8:ind+8+13])
        return datetime.fromtimestamp(timestamp/1000)
    
    def add_activity(self, activity_name, timestamp):
        
        if len(self.activities) == 0:
            self.activities.append({'name': activity_name, \
                                    'start': timestamp})
            self.events = [[]]
            return
        
        #Check if currently in activity
        if not activity_name == self.activities[-1]['name']:
            self.activities.append({'name': activity_name, \
                                    'start': timestamp})
            self.events.append([])
        else:
            self.activities[-1]['stop'] = timestamp

    
    def add_story_hear_event(self, timestamp, line):
        
        class_name = self.get_entry('class', line)
        
        if class_name == 'INFO':
            event_name = self.get_entry('name', line).lower()
            action_name = self.get_entry('action', line).lower()
            if len(event_name) > 0 or len(action_name) > 0:
                #Check if different from previous event
                if len(self.events[-1]) == 0 or \
                    not self.events[-1][-1]['event'] == event_name:
                    if action_name == 'play':
                        self.events[-1].append({'action': action_name, \
                                                'event': event_name, \
                                                'time': timestamp})
                    if event_name == 'next_step':
                        self.events[-1].append({'action': '', \
                                                'event': event_name, \
                                                'time': timestamp})
                    if event_name == 'wrong' or event_name == 'correct':
                         self.events[-1].append({'action': 'result', \
                                                'event': event_name, \
                                                'time': timestamp})
            
            backbutton_info = self.get_entry('BACKBUTTON', line).lower()
            if len(backbutton_info) > 0:
                self.activities[-1]['backbutton'] = True
    
    def add_story_echo_event(self, timestamp, line):
        
        class_name = self.get_entry('class', line)
        
        if class_name == 'INFO':
            event_name = self.get_entry('name', line).lower()
            action_name = self.get_entry('action', line).lower()
            if len(event_name) > 0 or len(action_name) > 0:
                #Check if different from previous event
                if len(self.events[-1]) == 0 or \
                    not self.events[-1][-1]['event'] == event_name:
                        
                    if action_name == 'play':
                        self.events[-1].append({'action': action_name, \
                                                'event': event_name, \
                                                'time': timestamp})
                    if event_name == 'next_step':
                        self.events[-1].append({'action': '', \
                                                'event': event_name, \
                                                'time': timestamp})
                    if event_name == 'wrong' or event_name == 'correct':
                         self.events[-1].append({'action': 'result', \
                                                'event': event_name, \
                                                'time': timestamp})
            
            backbutton_info = self.get_entry('BACKBUTTON', line).lower()
            if len(backbutton_info) > 0:
                self.activities[-1]['backbutton'] = True
                
    
    def add_activity_selector_event(self, timestamp, line):
        
        class_name = self.get_entry('class', line)
        
        if class_name == 'INFO':
            event_name = self.get_entry('name', line).lower()
            action_name = self.get_entry('action', line).lower()
            if len(self.events[-1]) == 0 or \
                        not self.events[-1][-1]['event'] == event_name:
                if action_name == 'play':
                    self.events[-1].append({'action': action_name, \
                                            'event': event_name, \
                                            'time': timestamp})
                    if event_name[:6] == 'it was':
                        if event_name[7:] == 'just right':
                            self.activities[-2]['feedback'] = 1
                        elif event_name[7:] == 'too hard':
                            self.activities[-2]['feedback'] = -1
                        elif event_name[7:] == 'too easy':
                            self.activities[-2]['feedback'] = 0
            
            backbutton_info = self.get_entry('BACKBUTTON', line).lower()
            if len(backbutton_info) > 0:
                self.activities[-1]['backbutton'] = True
    
    def add_write_word_event(self, timestamp, line):
        class_name = self.get_entry('class', line)
        if class_name == 'INFO':
            event_name = self.get_entry('name', line).lower()
            action_name = self.get_entry('action', line).lower()
            if len(self.events[-1]) == 0 or \
                        not self.events[-1][-1]['event'] == event_name:
                if action_name == 'play':
                    self.events[-1].append({'action': action_name, \
                                            'event': event_name, \
                                            'time': timestamp})
                    
                if event_name == 'enable input' or event_name == 'disable input':
                    self.events[-1].append({'action': '', \
                                            'event': event_name, \
                                            'time': timestamp})
            
            backbutton_info = self.get_entry('BACKBUTTON', line).lower()
            if len(backbutton_info) > 0:
                self.activities[-1]['backbutton'] = True
    
    def add_line(self, line):
        #Get the activity
        activity_name = self.get_entry('tutor',line)
        
        #Get timestamp
        timestamp = self.get_timestamp(line) 
        
        #Add activity
        self.add_activity(activity_name, timestamp)
        
        #Run function for specific activity
        if activity_name[:10] == 'story.hear':
            self.add_story_hear_event(timestamp, line)
        if activity_name[:10] == 'story.echo':
            self.add_story_echo_event(timestamp, line)
        if activity_name == 'activity_selector':
            self.add_activity_selector_event(timestamp, line)
        if activity_name[:9] == 'write.wrd':
            self.add_write_word_event(timestamp, line)
    
    def get_activity_timestamp(self, num):
        cur_num = 0
        
        for activity in self.activities:
            if activity['name'] == '<undefined>' or \
                activity['name'] == 'activity_selector':
                pass
            else:
                if cur_num == num:
                    return activity['start']
                else:
                    cur_num += 1
                    
