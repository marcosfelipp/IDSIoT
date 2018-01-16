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

    def set_count(self, count):
        '''
        :param count: number of connections to the same host as the current connection in the past two seconds 
        :return: None
        '''
        self.count = count

    def set_same_host_fetaures(self, serror_rate, rerror_rate, same_srv_rate, diff_srv_rate, srv_count):
        '''
        Function that set features - same-service connections
        :param serror_rate: % of connections that have ``SYN'' errors 
        :param rerror_rate: % of connections that have ``REJ'' errors 
        :param same_srv_rate: % of connections to the same service 
        :param diff_srv_rate: % of connections to different services 
        :param srv_count: number of connections to the same service as the current connection in the past two seconds 
        :return: None
        '''
        self.serror_rate = serror_rate
        self.rerror_rate = rerror_rate
        self.same_srv_rate = same_srv_rate
        self.diff_srv_rate = diff_srv_rate
        self.srv_count = srv_count

    def set_destination_features(self, srv_serror_rate, srv_rerror_rate, srv_diff_host_rate):
        '''
        Function that set features - same-host connections
        :param srv_serror_rate: % of connections that have ``SYN'' errors 
        :param srv_rerror_rate: % of connections that have ``REJ'' errors 
        :param srv_diff_host_rate: % of connections to different hosts 
        :return: None
        '''
        self.srv_rerror_rate = srv_rerror_rate
        self.srv_serror_rate = srv_serror_rate
        self.diff_srv_rate = srv_diff_host_rate