import os


def make_directories_as_necessary(filename):
    stub_pieces = filename.split('/')
    if len(stub_pieces) > 1:
        dir_so_far = stub_pieces[0]
        if dir_so_far != '' and not os.path.isdir(dir_so_far):
            os.makedirs(dir_so_far)
        for stub_piece in stub_pieces[1: -1]:
            dir_so_far += '/' + stub_piece
            if not os.path.isdir(dir_so_far):
                os.makedirs(dir_so_far)
        dir_so_far += '/' + stub_pieces[-1]
