import csv

class ActivityFeatures():
    
    def __init__(self, activity, video_filename):
        self.name = activity['name']
        self.start = activity['start']
        self.stop = activity['stop']
        self.feedback = activity['feedback']
        if 'backbutton' in activity.keys():
            self.backbutton = 'True'
        else:
            self.backbutton = 'False'
            
        self.video_filename = video_filename[:2]
        self.features = []
    
    def print_features(self, activity_ind):
        
        activity_name = self.name
        activity_name = activity_name.replace(':', '')
        activity_name = activity_name.replace('.', '')   
        
                
        filename = 'data/allfeature/' + self.video_filename[:2]+ '_' + \
                    str(activity_ind) + '_' + \
                    activity_name + '_allfeature.csv'
        
        f = open(filename, 'w', newline='')
        f_writer = csv.writer(f, delimiter=',')
        
        f_writer.writerow(['Video Filename', self.video_filename, 'Activity Ind', str(activity_ind), \
                           'Activity Name', self.name, 'Feedback', str(self.feedback), 'Backbutton', self.backbutton])
        f_writer.writerow(['Time', 'State', 'Head Proximity', 'Head Orientation', 'Gaze Direction', \
                           'Eye Aspect Ratio', 'Pupil Ratio', 'Lines', 'Picture Side'])
        for feature in self.features:
            if feature['sides'] == 0:
                side = 'left'
            else:
                side = 'right'
            if feature['head_prox'] > 10000:
                feature['head_prox']/= 100000.0
                
            f_writer.writerow([feature['time'], \
                    feature['state'], feature['head_prox'], \
                    feature['head_orient'], feature['gaze_dir'], \
                    feature['eye_aspect_ratio'], feature['pupil_ratio'], \
                    feature['lines'], side])
        
        f.close()
        print('Wrote to file:', filename)   
    
    
    def is_before(self, time1, time2):
        return time1 < time2

    
    def get_story_hear_features(self, events, vid_feat):
        
        features = []
        
        cur_state = 'listen'
        num_lines = 0
        
        event_id = 0
        line_ind_start = 0
        events_done = False
        
        for ii, vid_time in enumerate(vid_feat['timestamps']):
            if self.is_before(self.start, vid_time) and \
                self.is_before(vid_time, self.stop):
                feat = {}
                feat['time'] = (vid_time - self.start).total_seconds()
                feat['state'] = cur_state
                feat['head_prox'] = vid_feat['head_prox'][ii]
                feat['head_orient'] = vid_feat['head_orient'][ii]
                feat['gaze_dir'] = vid_feat['gaze_dir'][ii]
                feat['eye_aspect_ratio'] = vid_feat['eye_aspect_ratio'][ii]
                feat['pupil_ratio'] = vid_feat['pupil_ratio'][ii]
                feat['sides'] = vid_feat['sides'][ii]
                event_time = events[event_id]['time']
                if self.is_before(vid_time, event_time):
                    pass
                else:
                    if events[event_id]['action'] == 'play' and not events_done:
                        num_points = ii - line_ind_start
                        incr = 1.0/num_points
#                        print(incr, num_points, line_ind_start, ii)
                        for jj, kk in enumerate(range(line_ind_start, ii)):
                            features[kk]['lines'] += incr*jj
                        line_ind_start = ii
                        num_lines += 1
                    if event_id == len(events) - 1:
                        events_done = True
                    else:
                        event_id = min(event_id + 1, len(events)-1)
                feat['lines'] = num_lines

                features.append(feat)
        num_points = ii - line_ind_start
        incr = 1.0/num_points
        for jj, kk in enumerate(range(line_ind_start, ii+1)):
            features[kk]['lines'] += incr*jj
        return features
    
    def get_story_echo_features(self, events, vid_feat):
        
        features = []
        
        cur_state = 'listen'
        num_lines = 0
        
        event_id = 0
        line_ind_start = 0
        events_done = False
        
        for ii, vid_time in enumerate(vid_feat['timestamps']):
            if self.is_before(self.start, vid_time) and \
                self.is_before(vid_time, self.stop):
                feat = {}
                feat['time'] = (vid_time - self.start).total_seconds()
                feat['state'] = cur_state
                feat['head_prox'] = vid_feat['head_prox'][ii]
                feat['head_orient'] = vid_feat['head_orient'][ii]
                feat['gaze_dir'] = vid_feat['gaze_dir'][ii]
                feat['eye_aspect_ratio'] = vid_feat['eye_aspect_ratio'][ii]
                feat['pupil_ratio'] = vid_feat['pupil_ratio'][ii]
                feat['sides'] = vid_feat['sides'][ii]
                event_time = events[event_id]['time']
                if self.is_before(vid_time, event_time):
                    pass
                else:
                    if events[event_id]['action'] == 'play' and not events_done:
                        num_points = ii - line_ind_start
                        incr = 1.0/num_points
                        for jj, kk in enumerate(range(line_ind_start, ii)):
                            features[kk]['lines'] += incr*jj
                        
                        line_ind_start = ii
                        num_lines += 1
                        
                    if event_id == len(events) - 1:
                        events_done = True
                    else:
                        event_id = min(event_id + 1, len(events)-1)
                
                feat['lines'] = num_lines
                
                features.append(feat)
        num_points = ii - line_ind_start
        incr = 1.0/num_points
        for jj, kk in enumerate(range(line_ind_start, ii+1)):
            features[kk]['lines'] += incr*jj
        return features
    
    def get_write_word_features(self, events, vid_feat):
        
        features = []
        
        cur_state = 'listen'
        num_lines = 0
        
        event_id = 0
        line_ind_start = 0
        events_done = False

        for ii, vid_time in enumerate(vid_feat['timestamps']):
            if self.is_before(self.start, vid_time) and \
                self.is_before(vid_time, self.stop):
                feat = {}
                feat['time'] = (vid_time - self.start).total_seconds()
                feat['head_prox'] = vid_feat['head_prox'][ii]
                feat['head_orient'] = vid_feat['head_orient'][ii]
                feat['gaze_dir'] = vid_feat['gaze_dir'][ii]
                feat['eye_aspect_ratio'] = vid_feat['eye_aspect_ratio'][ii]
                feat['pupil_ratio'] = vid_feat['pupil_ratio'][ii]
                feat['sides'] = vid_feat['sides'][ii]
                event_time = events[event_id]['time']
                
                if self.is_before(vid_time, event_time):
                    pass
                else:
                    if events[event_id]['event'] == 'enable input':
                        cur_state = 'write'
                    if events[event_id]['event'] == 'disable input':
                        cur_state = 'listen'
                    
                    if events[event_id]['action'] == 'play' and not events_done:
                        if len(events[event_id]['event']) > 1 and \
                                not events[event_id]['event'] == \
                                'try to make it look more like this':
                            num_points = ii - line_ind_start
                            incr = 1.0/num_points
                            for jj, kk in enumerate(range(line_ind_start, ii)):
                                features[kk]['lines'] += incr*jj
                            
                            line_ind_start = ii
                            num_lines += 1
                            
                    if event_id == len(events) - 1:
                        events_done = True
                    else:
                        event_id = min(event_id + 1, len(events)-1)

                feat['lines'] = num_lines
                
                feat['state'] = cur_state
                features.append(feat)
        num_points = ii - line_ind_start
        incr = 1.0/num_points
        for jj, kk in enumerate(range(line_ind_start, ii+1)):
            features[kk]['lines'] += incr*jj
        return features
        
    
    def add_features(self, events, vid_feat):
        if len(vid_feat['timestamps']) == 0:
            return
        
        if self.name[:10] == 'story.hear':
            self.features = self.get_story_hear_features(events, vid_feat)
        elif self.name[:10] == 'story.echo':
            self.features = self.get_story_echo_features(events, vid_feat)
        elif self.name[:9] == 'write.wrd':
            self.features = self.get_write_word_features(events, vid_feat)
        else:
            print('Did not compute features for', self.name)
    
    