from ryu.lib import pcaplib

class PacketCollector:
    def __init__(self):
        # Create pcaplib.Writer instance with a file object
        # for the PCAP file
        self.pcap_writer = pcaplib.Writer(open('mypcap.pcap', 'wb'))

    def save_packet(self, packet):
        self.pcap_writer.write_pkt(packet)
