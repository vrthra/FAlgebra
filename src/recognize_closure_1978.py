import Recognize as R
import closure_1978 as Main

if __name__ == '__main__':
    R.main('./lang/js/grammar/javascript.fbjson', './lang/js/bugs/closure.1978.js', Main.my_predicate)
