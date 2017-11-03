import os
import numpy as np


class DataManipulation:
    def __init__(self):
        pass

    @staticmethod
    def read_file(dataset_name):
        '''
        :param dataset_name
        :return: 2 lists coitaining tuples and classification of attack
        '''

        matrix = []
        classes = []

        file_name = '../data/' + dataset_name + '.arff'

        training = os.path.join(os.path.dirname(__file__), file_name)

        services_type = {'aol': 1, 'auth': 2, 'bgp': 3, 'courier': 4, 'csnet_ns': 5, 'ctf': 6, 'daytime': 7,
                         'discard': 8, 'domain': 9,
                         'domain_u': 10, 'echo': 11, 'eco_i': 12, 'ecr_i': 13, 'efs': 14, 'exec': 15, 'finger': 16,
                         'ftp': 17, 'ftp_data': 18, 'gopher': 19,
                         'harvest': 20, 'hostnames': 21, 'http': 22, 'http_2784': 23, 'http_443': 24, 'http_8001': 25,
                         'imap4': 26, 'IRC': 27, 'iso_tsap': 28,
                         'klogin': 29, 'kshell': 30, 'ldap': 31, 'link': 32, 'login': 33, 'mtp': 34, 'name': 35,
                         'netbios_dgm': 36, 'netbios_ns': 37,
                         'netbios_ssn': 38, 'netstat': 39, 'nnsp': 40, 'nntp': 41, 'ntp_u': 42, 'other': 43,
                         'pm_dump': 44, 'pop_2': 45, 'pop_3': 46,
                         'printer': 47, 'private': 48, 'red_i': 49, 'remote_job': 50, 'rje': 51, 'shell': 52,
                         'smtp': 53, 'sql_net': 54, 'ssh': 55,
                         'sunrpc': 56, 'supdup': 57, 'systat': 58, 'telnet': 59, 'tftp_u': 60, 'tim_i': 61, 'time': 62,
                         'urh_i': 63, 'urp_i': 64, 'uucp': 65,
                         'uucp_path': 66, 'vmnet': 67, 'whois': 68, 'X11': 69, 'Z39_50': 70}

        protocol_type = {'tcp': 1, 'udp': 2, 'icmp': 3}

        flag = {'OTH': 1, 'REJ': 2, 'RSTO': 3, 'RSTOS0': 4, 'RSTR': 5, 'S0': 6, 'S1': 7, 'S2': 8, 'S3': 9, 'SF': 10,
                'SH': 11}

        # classes_atacks = {'back'            : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        #                   'buffer_overflow' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        #                   'ftp_write'       : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        #                   'guess_passwd'    : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        #                   'imap'            : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
        #                   'land'            : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        #                   'loadmodule'      : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        #                   'multihop'        : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        #                   'neptune'         : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                   'nmap'            : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        #                   'perl'            : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
        #                   'phf'             : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        #                   'pod'             : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'portsweep'       : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'rootkit'         : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'satan'           : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'smurf'           : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'teardrop'        : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'warezclient'     : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'warezmaster'     : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'ipsweep'         : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'spy'             : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'normal'          : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                   'unknown'         : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}

        classes_atacks = {'normal' : [0,1] , 'anomaly' : [1,0]}


        with open(training) as file:
            for line in file:
                tuple = line.strip().split(',')
                matrix.append(tuple)

        for tuple in matrix:
            tuple[1] = protocol_type.get(tuple[1])
            tuple[2] = services_type.get(tuple[2])
            tuple[3] = flag.get(tuple[3])

            vector = []
            var = classes_atacks.get(tuple[41])
            # if var == None:
            #     var = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

            vector.append(var)

            classes.append(vector)

            del tuple[41]
            # del tuple[41]

        for line in range(len(matrix)):
            for value in range(len(matrix[0])):
                matrix[line][value] = float(matrix[line][value])

        return matrix, classes
