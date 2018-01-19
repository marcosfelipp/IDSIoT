from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.ofproto import ofproto_v1_3
from ryu.controller.handler import set_ev_cls
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import CONFIG_DISPATCHER

from collector.statistics_request import StatsRequest13
from collector.rule_manager import RuleManager
from collector.packet_collector import PacketCollector

class OpenFlowController(app_manager.RyuApp):
    '''
    Class that communicate with switch
    '''
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    def __init__(self):
        super(OpenFlowController, self).__init__()
        self.datapath           = None
        self.statistics_request = None
        self.rule_manager       = None
        self.packet_collector   = None

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        self.datapath = ev.msg.datapath

        self.rule_manager = RuleManager(self)
        self.packet_collector = PacketCollector()

        self.statistics_request =  StatsRequest13(self, self.datapath)
        self.statistics_request.start()

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_hander(self, ev):
        self.rule_manager(ev)
        self.packet_collector.save_packet(ev.msg.data)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply)
    def stats_reply(self, ev):
        pass

    def send_flow_stats_request(self, message):
        '''
        Method to communicate with switch
        :return: None
        '''
        try:
            self.datapath.send_msg(message)
        except Exception, e:
            print("Error")
