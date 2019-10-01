import json
import os
import sys
from pip._internal import main as pip

import urllib.request as urllib


def get_last_release(name_package):
    urlpath = urllib.urlopen(os.path.join('https://pypi.org/pypi', name_package, 'json'))
    string = urlpath.read()

    wjdata = json.loads(string)
    releases = list(wjdata["releases"].keys())

    latest = releases[0]
    for release in releases:
        if wjdata["releases"][release][0]["upload_time"] > wjdata["releases"][latest][0]["upload_time"]:
            latest = release

    return latest


def install_latest_dev_package(name_package):
    latest_version = get_last_release(name_package)
    pip(['install', '--no-input', name_package + '==' + latest_version, '--upgrade'])


def main(argv=None):
    install_latest_dev_package(argv[1])


if __name__ == "__main__":
    main(sys.argv)
