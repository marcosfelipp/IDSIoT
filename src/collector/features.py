class Features:

    def __init__(self):
        self.duration                    = None
        self.protocol_type               = None
        self.service                     = None
        self.src_bytes                   = None
        self.dst_bytes                   = None
        self.flag                        = None
        self.land                        = None
        self.wrong_fragment              = None
        self.urgent                      = None
        self.time_based_traffic_features = None

    def set_time_based_traffic_features(self, time_based_traffic_features):
        '''
        :param time_based_traffic_features: Class that contain Traffic features computed using a two-second time window.
        :return: None
        '''
        self.time_based_traffic_features = time_based_traffic_features
