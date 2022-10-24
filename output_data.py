import requests
import json

#api_url = config['api_url']
#port = config['port']

#need to customise each of these, just not done it yet need caracal access.
#useful info on requests : https://requests.readthedocs.io/en/latest/user/quickstart/#passing-parameters-in-urls

def REST_on_liveliness_changed(port, api_url,
             team, alive_count, not_alive_count):
    myData = {"team": team,
              "alive_count": alive_count,
              "not_alive_count": not_alive_count,}
    requests.post(api_url + ':' + str(port) + '/on_liveliness_changed', json=myData)

def REST_on_requested_deadline_missed(port, api_url,
             team, missed_total_count):
    myData = {"team": team,
              "missed_total_count": missed_total_count}
    requests.post(api_url + ':' + str(port) + '/on_requested_deadline_missed', json=myData)

def REST_on_sample_lost(port, api_url,
             team, lost_reason):
    myData = {"team": team,
              "lost_reason": lost_reason}
    requests.post(api_url + ':' + str(port) + '/on_sample_lost', json=myData)

def REST_generic(port, api_url, team, tag):
    myData = {"team": team,
              "tag": tag}
    requests.post(api_url + ':' + str(port) + '/generic', json=myData)
