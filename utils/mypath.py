"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar10','cifar100', 'imagenet-r','iNature','imagenet10','imagenet100'}
        database=database.split("_")[0]
        assert(database in db_names)

        if database == 'cifar10':
            return '/inspurfs/group/hexm/image_classification_data/cifar10/'

        elif database == 'cifar100':
            return '/inspurfs/group/hexm/image_classification_data/cifar100/'

        elif database == 'imagenet-r':
            return '/inspurfs/group/hexm/image_classification_data/imagenet-r/'

        elif database == 'iNature':
            # return '/storage/data/zhangchy2/ncd_data/NCD/iNature2018'
            return '/inspurfs/group/hexm/image_classification_data/iNature2018/'
        elif database == 'imagenet':
            return "/storage/data/zhangchy2/ncd_data/NCD/"
        elif database == 'imagenet10':
            return "/storage/data/zhangchy2/ncd_data/NCD/imagenet10/"
        elif database == 'imagenet100':
            return "/storage/data/zhangchy2/ncd_data/NCD/imagenet100/"
        else:
            raise NotImplementedError
