import time
from threading import Thread

class StatsRequest13(Thread):

    def __init__(self,connection, sleep_time):
        Thread.__init__(self)

        self.sleep_time  = sleep_time
        self.connection  = connection
        self.close       = False


    def run(self):
        while not self.close:
            self.send_statistics_request()
            time.sleep(self.sleep_time)

    def send_statistics_request(self):
        try:
            ofp_parser = self.connection.ofproto_parser
            match = ofp_parser.OFPMatch()

            req = ofp_parser.OFPFlowStatsRequest(self.connection,match=match)
            self.driver.send_flow_stats_request(req)

        except Exception, e:
            print(e)