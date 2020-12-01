# -*- coding: utf-8 -*-
"""
@author: José Antonio Fiorote (jafiorote@gmail.com) 28/11/2020

@description: PyDash Project

Implementation of FDASH algorithm from paper "FDASH: A Fuzzy-Based MPEG/DASHAdaptation Algorithm"

"""

from player.parser import *
from player.player import *
from r2a.ir2a import IR2A
import time
import numpy as np


class R2AFDash(IR2A):

    def __init__(self, id):
        IR2A.__init__(self, id)
        self.parsed_mpd = ''
        self.qi = []
        self.d = 2  # seconds
        self.throughputs = []  # np array
        self.target_buff_t = 15  # seconds
        self.start_idx_br = -10
        self.curr_idx = 0

    def get_delta_t(self):

        """
        Get buffering time difference Δti of the last and previous one segments. Furthermore, get the difference of
        the two lasts Δti

        :return: Δti, ΔΔti
        """

        times = self.whiteboard.get_playback_segment_size_time_at_buffer()

        if len(times) > 3:
            delta_t = times[-1] - times[-2]
            diff_delta_t = (times[-1] - times[-2]) - (times[-3] - times[-4])
            return delta_t, diff_delta_t
        else:
            return 1, 0

    def buff_time(self, delta_t):

        """
        :param delta_t: buffering time difference Δti

        :return: list of tuples (a, b), where a is an integer flag and b is a float value.
        flags: 0 -> short, 1 -> close, 2 -> long.
        """

        if delta_t > 4 * self.target_buff_t:
            return [(2, 1)]

        elif delta_t > self.target_buff_t:
            v1 = 1 - (1 / ((self.target_buff_t * 3) * (delta_t - self.target_buff_t)))
            v2 = 1 / ((self.target_buff_t * 3) * (delta_t - self.target_buff_t))
            return [(1, v1), (2, v2)]

        elif delta_t > self.target_buff_t / 3:
            v0 = 1 - (1 / ((self.target_buff_t / 3) * (delta_t - 2 * self.target_buff_t / 3)))
            v1 = 1 / ((self.target_buff_t / 3) * (delta_t - 2 * self.target_buff_t / 3))
            return [(0, v0), (1, v1)]

        else:
            return [(0, 1)]

    def evol_buff_time(self, diff_delta_t):

        """
        :param diff_delta_t: buffering time difference difference ΔΔti

        :return: list of tuples (a, b), where a is an integer flag and b is a float value.
        flags: 0 -> falling, 1 -> steady, 2 -> rising.
        """

        if diff_delta_t > 4 * self.target_buff_t:
            return [(2, 1)]

        elif diff_delta_t > 0:
            v1 = 1 - (1 / (self.target_buff_t * 4) * diff_delta_t)
            v2 = 1 / (self.target_buff_t * 4) * diff_delta_t
            return [(1, v1), (2, v2)]

        elif diff_delta_t > - 2 * self.target_buff_t / 3:
            v0 = 1 - (1 / ((self.target_buff_t / 3 * 2) * (diff_delta_t + 2 * self.target_buff_t / 3)))
            v1 = 1 / (self.target_buff_t / 3 * 2) * (diff_delta_t + 2 * self.target_buff_t / 3)
            return [(0, v0), (1, v1)]

        else:
            return [(0, 1)]

    def get_rules(self, buff_time_vec, buff_evol_vec):

        """
        Fuzzy if-then rules controller

        :param buff_time_vec: list of tuples with short, close, and long flags, and value of buffering time
        :param buff_evol_vec: list of tuples with falling, steady, rising flags and value of evolution in buffering time
        :return: Nine elements array with fuzzy rules results
        """

        # R, SR, NC, SR, NC, SI, NC, SI, I
        rules = np.zeros(9, dtype=float)

        for i in buff_time_vec:
            for j in buff_evol_vec:

                value = np.amin([i[1], j[1]])

                if i[0] == 0 and j[0] == 0:
                    rules[0] = value

                elif i[0] == 1 and j[0] == 0:
                    rules[1] = value

                elif i[0] == 2 and j[0] == 0:
                    rules[2] = value

                elif i[0] == 0 and j[0] == 1:
                    rules[3] = value

                elif i[0] == 1 and j[0] == 1:
                    rules[4] = value

                elif i[0] == 2 and j[0] == 1:
                    rules[5] = value

                elif i[0] == 0 and j[0] == 2:
                    rules[6] = value

                elif i[0] == 1 and j[0] == 2:
                    rules[7] = value

                elif i[0] == 2 and j[0] == 2:
                    rules[8] = value

        return rules

    def f_method(self, rules):

        """
        Centroid method.

        :param rules: array with fuzzy rules results
        :return: increase/decrease factor f
        """

        # parameters:
        n1 = 0.5
        n2 = 0.25
        z = 1
        p1 = 1.5
        p2 = 2

        i = rules[8]
        si = np.sqrt(np.power(rules[5], 2) + np.power(rules[7], 2))
        nc = np.sqrt(np.power(rules[2], 2) + np.power(rules[4], 2) + np.power(rules[6], 2))
        sr = np.sqrt(np.power(rules[1], 2) + np.power(rules[3], 2))
        r = rules[0]

        f = ((n2 * r) + (n1 * sr) + (z * nc) + (p1 * si) + (p2 * i)) / (sr + r + nc + si + i)

        return f

    def get_rd(self):

        """
        Calculate throughput mean for k segments for d seconds

        :return: throughput mean rd
        """

        throughputs = []
        curr_time = time.perf_counter()
        for row in self.throughputs:
            if row[0] > curr_time - self.d:
                throughputs.append(row[-1])

        return np.mean(throughputs)

    def new_bit_rate_idx(self, f):

        """

        :param f: increase/decrease factor f
        :return: index for require next segment
        """

        new_idx = 0

        if len(self.throughputs) < 1:
            new_idx = self.start_idx_br
            self.curr_idx = new_idx

        bi = f * self.get_rd()

        for idx in range(len(self.qi)):
            if bi >= self.qi[(idx + 1) * -1]:
                new_idx = (idx + 1) * -1
                break
            else:
                new_idx = -len(self.qi)

        buffer_sz = self.whiteboard.get_playback_buffer_size()[-1][1] if \
            len(self.whiteboard.get_playback_buffer_size()) else 1
        t_compare = buffer_sz + (self.get_rd() / self.qi[new_idx] - 1) * self.d

        if new_idx > self.curr_idx and self.target_buff_t > t_compare:
            new_idx = self.curr_idx
        elif new_idx < self.curr_idx and self.target_buff_t < t_compare:
            new_idx = self.curr_idx
        else:
            self.curr_idx = new_idx


        return new_idx

    def handle_xml_request(self, msg):
        self.throughputs = np.zeros((1, 3), dtype=float)
        self.throughputs[0][0] = time.perf_counter()
        self.send_down(msg)

    def handle_xml_response(self, msg):
        # getting qi list
        self.parsed_mpd = parse_mpd(msg.get_payload())
        self.qi = self.parsed_mpd.get_qi()

        self.send_up(msg)

    def handle_segment_size_request(self, msg):
        # time to define the segment quality choose to make the request

        delta_t, diff_delta_t = self.get_delta_t()
        buff_time_vec = self.buff_time(delta_t)
        buff_evol_vec = self.evol_buff_time(diff_delta_t)
        rules = self.get_rules(buff_time_vec, buff_evol_vec)
        f = self.f_method(rules)
        idx = self.new_bit_rate_idx(f)
        msg.add_quality_id(self.qi[idx])
        throughputs = np.zeros((1, 3), dtype=float)
        self.throughputs = np.vstack((self.throughputs, throughputs))
        self.throughputs[-1][0] = time.perf_counter()
        self.send_down(msg)

    def handle_segment_size_response(self, msg):
        self.throughputs[-1][1] = time.perf_counter()
        self.throughputs[-1][2] = msg.get_bit_length() / (self.throughputs[-1][1] - self.throughputs[-1][0])
        self.send_up(msg)

    def initialize(self):
        print(">>>>>>>>>>>>>>>> Starting FDASH...")

    def finalization(self):
        print(">>>>>>>>>>>>>>>> Ending FDASH")