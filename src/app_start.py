from ryu.ofproto import ofproto_v1_3
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller import ofp_handler
from ryu.controller.handler import set_ev_cls
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import CONFIG_DISPATCHER

class AppStart(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    def __init__(self):
        super(AppStart, self).__init__()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        pass

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_hander(self, ev):
        pass

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply)
    def stats_reply(self, ev):
        pass



