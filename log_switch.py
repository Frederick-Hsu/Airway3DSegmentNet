#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : log_switch.py
# Brief     : To turn ON/OFF the log message printing
#
#


import logging

# Objects ==========================================================================================
log = logging.Logger(__name__)
# changing from logging.DEBUG to logging.ERROR, it can turn off all log.info(), log.warning() statements
log.setLevel(logging.DEBUG)
log.warning("Segment the pulmonary 3D airway model")

