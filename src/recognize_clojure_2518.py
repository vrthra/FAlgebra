import Recognize as R
import clojure_2518 as Main

if __name__ == '__main__':
    R.main('./lang/clojure/grammar/clojure.fbjson', './lang/clojure/bugs/clj-2518.clj', Main.my_predicate)
