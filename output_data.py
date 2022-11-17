import requests
import json

# api_url = config['api_url']
# port = config['port']

# need to customise each of these, just not done it yet need caracal access.
# useful info on requests : https://requests.readthedocs.io/en/latest/user/quickstart/#passing-parameters-in-urls

# may need to create an adapter to be able to set ports - as defined this may not currently work
# https://stackoverflow.com/questions/47202790/python-requests-how-to-specify-port-for-outgoing-traffic

def post_error(error_type, predicted_timestamp):
    port = "8081"
    url = "192.168.200.171"
    team = 1
    if error_type == 0:
        post_on_liveliness_changed(port, url, team, predicted_timestamp)
    elif error_type == 1:
        post_on_requested_deadline_missed(port, url, team, predicted_timestamp)
    elif error_type == 2:
        post_on_sample_lost(port, url, team, predicted_timestamp)
    else:
        post_generic(port, url, team, predicted_timestamp)


def post_on_liveliness_changed(port, api_url, team, timestamp, alive_count=0, not_alive_count=0):
    myData = {"team": team,
              "timestamp": timestamp,
              "alive_count": alive_count,
              "not_alive_count": not_alive_count}
    requests.post(api_url + ":" + port + '/on_liveliness_changed', json=myData)


def post_on_requested_deadline_missed(port, api_url, team, timestamp, missed_total_count=0):
    myData = {"team": team,
              "timestamp": timestamp,
              "missed_total_count": missed_total_count}
    requests.post(api_url + ":" + port + '/on_requested_deadline_missed', json=myData)


def post_on_sample_lost(port, api_url, team, timestamp, lost_reason="None"):
    myData = {"team": team,
              "timestamp": timestamp,
              "lost_reason": lost_reason}
    requests.post(api_url + ":" + port + '/on_sample_lost', json=myData)


def post_generic(port, api_url, team, timestamp, tag="ERROR"):
    myData = {"team": team,
              "timestamp": timestamp,
              "tag": tag}
    requests.post(api_url + ":" + port + '/generic', json=myData)
