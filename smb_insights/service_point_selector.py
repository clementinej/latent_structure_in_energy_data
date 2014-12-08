from collections import OrderedDict

from IPython.html.widgets import (SelectWidget, interactive)

from .schedules import (get_service_points, get_day_values, find_open_close, find_threshold, plot_day,
                        usage_when_closed, calculate_waste, day_or_night, times_of_day)



class ServicePointSelector:
    """
    Driver for interactive plots SMB service points
    """
    def __init__(self, quarter_hour_data):
        self.qh = quarter_hour_data
        self.service_points = sorted(get_service_points(self.qh))

    def interactive_selector(self, SMB):
        '''
        Helper method called by ``self.run_selector`` which allows plotting for a specific model and customer.
        :param SMB: Int util-service-point-id to analyze
        :return: Nothing. Called for plotting side-effect.
        '''
        print "SMB: {spid}".format(spid=SMB)
        day_values = get_day_values(self.qh, SMB)
        total_daily_usage = day_values.sum()
        (open_index, closed_index) = find_open_close(day_values)
        times = times_of_day()
        print ("{btype} business -- opens: {open}, closes: {close}"
               .format(btype=day_or_night(open_index, closed_index), open=times[open_index], close=times[closed_index]))
        plot_day(day_values, open_index, closed_index)
        closed_usage = usage_when_closed(day_values, open_index, closed_index)
        waste_usage = calculate_waste(day_values, open_index, closed_index)
        print ("Closed usage = %0.1f kWh, which is %0.1f%% of total daily usage" %
               (closed_usage, closed_usage/total_daily_usage*100))
        print ("Waste usage = %0.1f kWh, which is %0.1f%% of total daily usage" %
               (waste_usage, waste_usage/total_daily_usage*100))



    def run_selector(self):
        '''
        Driver of interactive plotting of SMB util-service-point-id
        :return: Container widget with child indicating which util-service-point to plot
        '''
        smb_selection = SelectWidget(values=OrderedDict((str(id), id) for id in self.service_points))
        i = interactive(self.interactive_selector, SMB=smb_selection)
        return i