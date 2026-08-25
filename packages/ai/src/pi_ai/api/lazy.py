def lazy_import(name):
    return __import__(name, fromlist=['*'])

