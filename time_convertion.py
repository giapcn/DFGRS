'''
    created by Giap
    using to calculate time effect used in GRS
'''
import datetime as dt
import math
class time_convertion:
    def __init__(self, Root=0, Frequency=86400, Lamda=0.001):
        # frequency = 86400 -> conver time to unit by day
        # lamda = 0.1, control the effect of time (time decay), lamda increases cause higher effect
        self.root= Root
        self.frequency = Frequency # window size
        self.lamda=Lamda

    def set_root(self, Root):
        self.root = Root

    def set_frequency(self, Frequency):
        self.frequency = Frequency

    def set_lamda(self, Lamda):
        self.lamda

    def time_transpose(self, Timepoint):
        timeplot = (Timepoint-self.root)/self.frequency
        return math.floor(timeplot)

    # using for real time point (according to datetime library)
    def realtime_distance(self, Timepoint1, Timepoint2):
        timeplot1 = math.floor((Timepoint1-self.root)/self.frequency)
        timeplot2 = math.floor((Timepoint2 - self.root) / self.frequency)
        return abs(timeplot2-timeplot1)

    # time distance according to time decay set up
    def time_distance(self, Timepoint1, Timepoint2):
        return abs(Timepoint2-Timepoint1)

    # time decay bases on distance according to time points
    def time_effect(self, Timepoint1, Timepoint2):
        return 1/(1+self.lamda*abs(Timepoint2 - Timepoint1))