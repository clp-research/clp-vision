# coding: utf-8
from __future__ import division

import argparse
import yaml

import sys
sys.path.append('../Utils')


class TaskFunctions(object):
    def tsk_saiapr(self, config):
        print "saiapr", config

    def exec_task(self, task, config):
        fn = getattr(self, 'tsk_' + task, None)
        if fn is None:
            print '%s is an unkown task' % task
            sys.exit(1)
        else:
            fn(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess corpora for DSG vision/language experiments')
    parser.add_argument('-c', '--config_file',
                        help='''
                        path to config file with data paths.
                        default: '../Config/default.yaml' ''',
                        default='../Config/default.yaml')
    parser.add_argument('task',
                        help='''
                        task to do. One of: 'saiapr'
                        ''')
    args = parser.parse_args()

    tfs = TaskFunctions()

    try:
        with open(args.config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    except IOError:
        print 'no config file found at %s' % (args.config_file)
        sys.exit(1)

    # print dir(tfs)

    # TODO: 'all' task, runs all tsk_ functions

    tfs.exec_task(args.task, cfg)
