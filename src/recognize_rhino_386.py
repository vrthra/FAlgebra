import Recognize as R
import rhino_386 as Main

import sys
if __name__ == '__main__':
    R.main('./lang/js/grammar/javascript.fbjson', './lang/js/bugs/rhino.386.js', Main.my_predicate)
