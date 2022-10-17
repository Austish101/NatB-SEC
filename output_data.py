import requests


def api_post(url, port, error_type, team, alive_count, not_alive_count, lost_reason, missed_total_count, tag):
    post_data = {"team": team,
                 "error_type": error_type,
                 "alive_count": alive_count,
                 "not_alive_count": not_alive_count,
                 "lost_reason": lost_reason,
                 "missed_total_count": missed_total_count}

    post_url = "%s:%s" % (url, port)

    requests.post(post_url, json=post_data)
