class TimeBasedTrafficFeatures:
    '''
    Traffic features computed using a two-second time window.
    '''
    def __init__(self):
        self.count              = None

        # Same host features:
        self.serror_rate        = None
        self.rerror_rate        = None
        self.same_srv_rate      = None
        self.diff_srv_rate      = None
        self.srv_count          = None

        # Same destination features:
        self.srv_serror_rate    = None
        self.srv_rerror_rate    = None
        self.srv_diff_host_rate = None