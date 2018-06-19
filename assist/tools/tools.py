'''@file tools.py
contains some usefull tools for the scripts'''

import os
import shutil
from ConfigParser import ConfigParser

def symlink(source, link_name):
    '''
    create a symlink, if target exists remove

    args:
        source: the file the link will be pointing to
        link_name: the path to the link file
    '''

    if os.path.exists(link_name):
        os.remove(link_name)

    os.symlink(source, link_name)


def safecopy(src, dst):
    '''only copy src to dest if dst does not exits'''

    if not os.path.exists(dst):
        shutil.copyfile(src, dst)

def writefile(filename, strings):
    '''write a dictionary of strings to a file'''

    if not os.path.exists(filename):
        with open(filename, 'w') as fid:
            for name in strings:
                fid.write('%s %s\n' % (name, strings[name]))

def default_conf(conf, default_path):
    '''put the defaults in the configuration if it is not defined

    args:
        conf: the conf as a ConfigParser object
        default_path: the path to the default location
    '''

    #read the default conf
    default = ConfigParser()
    default.read(default_path)

    #write the default values that are not in the config to the config
    for section in default.sections():
        if not conf.has_section(section):
            conf.add_section(section)
        for option in default.options(section):
            if not conf.has_option(section, option):
                conf.set(section, option, default.get(section, option))
