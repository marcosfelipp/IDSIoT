import time
from threading import Thread

class StatsRequest13(Thread):

    def __init__(self, of_controller, datapath):
        Thread.__init__(self)

        self.sleep_time     = 2 # Time to send statistics request message to the switch
        self.datapath       = datapath
        self.of_controller  = of_controller
        self.close          = False

    def run(self):
        while not self.close:
            self.send_statistics_request()
            time.sleep(self.sleep_time)


    def create_stats_same_service(self):
        pass

    def create_stats_same_host(self):
        pass

    def send_statistics_request(self, match):
        try:
            ofp_parser = self.datapath.ofproto_parser
            match = ofp_parser.OFPMatch()

            req = ofp_parser.OFPFlowStatsRequest(self.datapath, match=match)
            self.of_controller.send_flow_stats_request(req)

        except Exception, e:
            print(e)