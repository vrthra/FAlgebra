#!/usr/bin/env python
# coding: utf-8

# # Algebra of Failure Inducing Input Patterns

# ## What

# We are trying to generate grammars that produce inputs that contain or does not specific behaviors (or complex combinations thereof) so, one can say things of the sort: I want to generate inputs with parenthesis, but discard everything with doubled parenthesis.
# 
# While boolean grammars are the formalism for such things (e.g A & B & !C), and one can trivially construct multi-level recognizers for it (parse A, parse B, not parse C), generation is much harder, and as far as I can see, no one has found a way to easily generate inputs from such expressions (other than the generate and filter approach, which is really inefficient). Further, the boolean grammars are already beyond context free grammars, and our current techniques such as annotation of probabilities, and feedback on grammar nodes will not work on them. What we do here, is to recognize that for fault pattern additions and removal, one can stick to a context-free subset, and one can guarantee the properties of the resulting grammar.
# 
# Other things we can likely do: Generate grammars that produce a given prefix or a given suffix, and combinations of those.

# ## Why

# It is often necessary to include multiple patterns in an input to trigger a fault. For example, a certain fault may occur only if a particular element is seen in advance, or a fault induced early on may only be triggered by a later element. Similarly, one may want to specify that a particular code element is covered, as well as a particular input element be/not be present (e.g. on bugfixes). Finally, one may also want to avoid triggering bugs that are already known.

# ## How
# 
# We first extract the abstract syntactical patterns that correspond to given behaviors, and then generate a refinement grammar from the original grammar that follows the algebraic specifications.
# 
# The main question being asked is, how to combine and negate fault patterns.
# 
# We use `~F` for a nonterminal that is guaranteed not to contain a given fault, and `+F` for a nonterminal that is guaranteed to contain at least one instance of the given fault.

# ## How is it done?

# We start with the concept of an _abstract pattern_ form `DDSet`.
# 
# ### Abstract Pattern
# 
# An _abstract pattern_ is a parse tree of a fault inducing input such that all non-causal subtrees are marked abstract. From an _abstract pattern_ we derive its _charecteristic node_.
# 
# In essence, an abstract pattern is a parse tree with typed holes.
# 
# #### Charecteristic node
# 
# A _charecteristic node_ of an _abstract pattern_ is the smallest subtree (and the corresponding node) that completely captures the concrete _terminal_ symbols in the _abstract pattern_.
# 
# #### Linear grammar of a parse tree
# 
# A _linear grammar_ of a parse tree is a grammar such that the grammar can produce only the given parse tree. It is constructed by marking each node in the parse tree with a unique suffix, and extracting the grammar of the suffixed tree. The _linear grammar_ of the _charecteristic node_ will produce the _exact_ string that produced the fault. The start symbol of the grammar is the nonterminal of the characteristic node. The linear grammar is converted to pattern grammar by marking abstract various nodes.
# 
# #### Abstract pattern grammar
# 
# This is derived from the _linear grammar_ of the _characteristic node_ where the nonterminals in the grammar that correspond to abstract nodes are replaced by the non-suffixed general nonterminals in the original grammar. The start symbol of the abstract pattern grammar remains the same as the linear grammar of the characteristic node. The important thing to note here is that if one wishes to reproduce the given fault, the parse tree __should contain__ the nonterminals of the linear grammar.
# 
# #### Negated abstract pattern grammar
# 
# The negated abstract pattern grammar for a fault is constructed from the abstract pattern grammar of that fault by replacing the definitions one token at a time with a nonterminal guaranteed not to match the particular expansion in the pattern grammar. To this set of rules is added the definitions from the original grammar that did not match the pattern grammar. The idea is to essentially form a _conjunction_ of possible negations.
# 
# 
# ```
# <X> := <A> <B>
# <A> := <D>
#      | a <C>
# <B> := B
#      | b
# <C> := C
#      | C
# <D> := d
# ```
# Say the pattern grammar is:
# ```
# <X> := <A_1> <B_2>
# <A_1> := a <C_3>
# <B_2> := b
# <C_3> := c
# ```
# 
# Then, the negated pattern grammar is (we define it using the symbol `^F` to avoid confusing with `+F`) --- The difference is that `+F` guarantees the full fault, while `^F` represents a partial fault.
# ```
# <^X> := <^A_1> <B_2>
#       | <A_1> <^B_2>
# <A_1> := a <C_3>
#        | <D*>
# <^A_1> := a <^C_3>
#         | A
# <B_2> := b
# <^B_2> := B
# <C_3> := c
# <^C_3> := C
# ```
# 
# The `<D*>` here is simply `<D>` if `<D>` cannot reach the faulty key `<X>`. If it can, then it is a negation `<D-X>`.
# 
# If a particular nonterminal cannot be negated, then the rules that use that negated nonterminal is removed, and the nonterminals that have no rules are removed recursively.

# #### Grammar with the guarantee that at least one instance of the fault will always be present (G+).
# 
# We mark nonterminals that are guaranteed to contain at least a given single fault as `+F` and those nodes that are guaranteed to not contain that fault as `~F` where `<F>` is the original nonterminal.
# 
# Given a grammar that starts with 
# 
# ```
# <start> := <A> <B> <C>
#          ...
# ```
#     
# The grammar will contain at least a single fault if that single rule is replaced by this set of rules
# 
# ```
# <1start> := <+A> <B> <C>
#          | <A> <+B> <C>
#          | <A> <B> <+C>
# ...
# ```
# Similarly, if one assumes that the nonterminal `<A>` can contain faults, and it has the following rules (nonterminals indicated by `<..>`)
# ```
# <A> := x <P> <P> x
#      | x <P> x
# ```
# then, the following definition is guaranteed to produce at least one instance of the fault in any expansion
# ```
# <1A> := x <+P> <P> x
#      | x <P> <+P> x
#      | x <+P> x
# ```

# #### Grammar with the guarantee that no instance of the fault will be present (G-)
# 
# Negation is simply negation of all nonterminals
# ```
# <~A> := x <~P> <~P>
#       | x <~P>
# ```

# #### Grammar with the guarantee that at most one instance of the fault will be present (G*)
# 
# ```
# <1start> := <A> <~B> <~C>
#          | <~A> <B> <~C>
#          | <~A> <~B> <C>
# ...
# ```

# #### Grammar with the guarantee that exactly one instance of the fault will be present
# 
# ```
# G_1 = G+ & G*
# ```

# #### Base cases
# 
# There are two base cases for this recursion. The first is when the nonterminal is the charecteristic node of the fault being inserted. In this case, the definition `<+F>` is simply the start symbol of the abstract pattern grammar (and the abstract pattern grammar is merged into the grammar). Similarly with the negation terminal.
# 
# The second base case is when the nonterminal cannot produce the fault (e.g `<digit>` for producing a parenthesis). Here, the solution is simple. The nonterminal `<+F>` is defined as empty, and the rules that use that nonterminal is removed, and the nonterminals with empty rules are removed recursively as before. The negation `<~F>` is simply `<F>` because it is guaranteed not to produce a fault.

# ### (|) two refined grammars.
# 
# We generate grammars with the faults on boths sides. Next, we merge the grammars. Next, we remove any rule from the definition that is more refined than another rule in the same definition. A rule is more refined than another rule if 1) they both were derived from the same original rule from the original grammar (checked by looking at the stems and terminals) and 2) for any given token, the token at the same position of the other rule has a superset expansion. E.g a digit with expansion `[2, 4]` is a refinement of a digit with expansion `[1, 2, 3, 4, 5]`. (The fixpoint is computed). (This means that the disjunction of the original grammar with any fault inducing grammar will always be the original grammar.)

# ### (&) two refined grammars.
# 
# Conjunction of two refined grammars is simply the conjunction of both start symbols. The conjunction of two matching (same stem) nonterminals is a conjunction of their matching rules (match with the same terminals and stem of nonterminals). Conjunction of two rules is single rule with each nonterminal representing a conjunction of two corresponding nonterminals at the corresponding places from the two rules. When there are multiple matching rules, the rules produced are all pairs.
# 
# Note that one can generate an `atleast one fault grammar` (by skipping negation in the exactly one fault grammar) and an `atmost one fault grammar` (by skipping fault insertion in the exactly one fault grammar), and generate the `exactly one fault grammar` by generating a conjunction of both.

# ## Limitations

# The limitations are as follows:
# 1. Only applicable to deterministic base grammars (at least deterministic parts of the grammar). The grammar is further restricted (without losing generality) to have the start symbol correspond to a single rule such that the rule contains only a single nonterminal (and any number of terminals).
# 2. The fault patterns are neither sound nor complete w.r.t the failure (this is a limitation of the ddset fault pattern).
#    i.e:
#    The same failure may result from different patterns. Hence, negating one pattern does not mean negating that failure fully (not complete)
#    The same pattern in a different context may not result in a failure (not sound)
# 4. Adding new fault patterns results in almost exponential increase (worst case) in the grammar rules of corresponding non terminals.
# 5. We assume that rules for a nonterminal are non-redundant for the base grammar (if not, it becomes nondeterministic).

# The faults can from several different inputs. The idea is that the characterizing node, and abstraction removes the influences of the specific parse tree.

# ## Magick

# We start with a few Jupyter magics that let us specify examples inline, that can be turned off if needed for faster execution. Switch TOP to False if you do not want examples to complete.

# In[1]:


TOP = True #__name__ == '__main__'


# In[2]:


from IPython.core.magic import  (Magics, magics_class, cell_magic, line_magic, line_cell_magic)
class B(dict):
    def __getattr__(self, name):
        return self.__getitem__(name)
@magics_class
class MyMagics(Magics):
    def __init__(self, shell=None,  **kwargs):
        super().__init__(shell=shell, **kwargs)
        self._vars = B()
        shell.user_ns['VARS'] = self._vars

    @cell_magic
    def var(self, line, cell):
        self._vars[line.strip()] = cell.strip()
 
    @line_cell_magic
    def top(self, line, cell=None):
        if TOP:
            if cell is None:
                cell = line
            ip = get_ipython()
            res = ip.run_cell(cell)

get_ipython().register_magics(MyMagics)


# In[3]:


EXCEPTION_HAPPENED = False


# In[4]:


from IPython.core.ultratb import AutoFormattedTB
itb = AutoFormattedTB(mode = 'Plain', tb_offset = 1)
def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    global EXCEPTION_HAPPENED
    EXCEPTION_HAPPENED = True
    # Show the error within the notebook, don't just swallow it
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
    # grab the traceback and make it into a list of strings
    #stb = itb.structured_traceback(etype, evalue, tb)
    #sstb = itb.stb2text(stb)
get_ipython().set_custom_exc((Exception,), custom_exc)


# In[5]:


import sys


# In[6]:


get_ipython().run_line_magic('top', 'assert sys.version_info[0:2] == (3, 7)')


# In[7]:


# !pip3 install sympy


# ## Fault Patterns

# ### Grammar

# A context-free grammar is represented as a Python dict, with each nonterminal symbol forming a key, and each nonterminal _defined_ by a list of expansion rules. For example, the expression grammar for parsing arithmetic expressions is given below.

# #### Definitions
# 
# ##### String
# 
# A string is composed of zero or more _alphabets_.
# 
# E.g $a b c d$
# 
# ##### Alphabet
# 
# An alphabet is a single unit in any derived string.
# 
# E.g $a$
# 
# ##### Terminal symbol
# 
# A terminal symbol is composed of _one or more_ alphabets.
# 
# E.g. $a b$
# 
# ##### Nonterminal symbol
# 
# A nonterminal symbol is an abstraction that is defined elsewhere in the grammar.
# E.g. $X_{nt}$
# 
# ##### Rule
# 
# A rule is composed of a sequence of terminal or nonterminal symbols.
# 
# E.g. $A^{rule} = X_{nt} \cdot Y_{t} \cdot Z_{nt}$
# 
# 
# ##### Definition
# 
# Each nonterminal symbol in the grammar corresponds to a _definition_ of that symbol. The definition consists of one or more alternative _rules_. (e.g $X_{nt} \models A^{rule} \lor B^{rule} \lor C^{rule}$).
# 
# ##### Context Free Grammar
# 
# A context free grammar is composed of a set of nonterminals and their corresponding definitions.
# 
# E.g. $ G = \{\{ X_{nt} \models A^{rule} \lor B^{rule} \lor C^{rule}\} \land \{Y_{nt} \models P^{rule} \lor Q^{rule}\} \} $

# Here is a simple context free grammar for Boolean expressions

# In[8]:


import string


# In[9]:


BEXPR_GRAMMAR = {
    '<start>': [['<bexpr>']],
    '<bexpr>': [
        ['<bop>', '(', '<bexpr>', ',', '<bexpr>', ')'],
        ['<bop>',  '(', '<bexpr>', ',', '<bexpr>', ')'],
        ['<bop>', '(', '<bexpr>', ')'],
        ['<fault>']],
    '<bop>' : [['and'], ['or'], ['neg']],
    '<fault>': [['<letters>'], []],
    '<letters>': [
        ['<letter>'],
        ['<letter>', '<letters>']],
    '<letter>': [[i] for i in (string.ascii_lowercase + string.ascii_uppercase + string.digits) + '_+*.-']
}


# In[10]:


BEXPR_START = '<start>'


# An example grammar for arithmetic expressions.

# In[11]:


EXPR_GRAMMAR = {'<start>': [['<expr>']],
 '<expr>': [['<term>', ' + ', '<expr>'],
  ['<term>', ' - ', '<expr>'],
  ['<term>']],
 '<term>': [['<factor>', ' * ', '<term>'],
  ['<factor>', ' / ', '<term>'],
  ['<factor>']],
 '<factor>': [['+', '<factor>'],
  ['-', '<factor>'],
  ['(', '<expr>', ')'],
  ['<integer>', '.', '<integer>'],
  ['<integer>']],
 '<integer>': [['<digit>', '<integer>'], ['<digit>']],
 '<digit>': [['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9']]}


# In[12]:


EXPR_START = '<start>'


# We define an input from which we extract our patterns. Note that we do not use the predicate; rather we assume that we already have a few such predicates.

# In[13]:


#%%top
expr_input =  '1 + ((2 * 3 / 4))'
bexpr_input = 'and(or(neg(A)),or(B))'


# Note the convetion we used: Each nonterminal is enclosed in angle brackets. E.g. `<expr>`. We now define a function that can distinguish terminal symbols from nonterminals.

# The `is_nt()` function checks if the given node is a terminal or not.

# In[14]:


def is_nt(symbol):
     return symbol and (symbol[0], symbol[-1]) == ('<', '>')


# #### The Parser

# Given the grammar, and an input, we can parse it into a derivation tree.
# The `Parser` below is from [fuzzingbook.org](https://www.fuzzingbook.org/html/Parser.html), and provides a generic context-free parser. This is present in the `src` directory.

# In[15]:


from Parser import EarleyParser as Parser


# How do we check that our parse succeeded? We can convert the derivation tree back to the original string and check for equality.

# The `tree_to_str()` function converts a derivation tree to its original string. We use a non recursive definition.

# In[16]:


def tree_to_str(tree):
    expanded = []
    to_expand = [tree]
    while to_expand:
        (key, children, *rest), *to_expand = to_expand
        if is_nt(key):
            #assert children # not necessary
            to_expand = list(children) + list(to_expand)
        else:
            assert not children
            expanded.append(key)
    return ''.join(expanded)


# In[17]:


Ts = tree_to_str


# In[18]:


#%%top
expr_parser = Parser(EXPR_GRAMMAR, start_symbol=EXPR_START, canonical=True)
expr_tree = list(expr_parser.parse(expr_input))[0]


# In[19]:


get_ipython().run_line_magic('top', "assert tree_to_str(expr_tree) == '1 + ((2 * 3 / 4))'")


# We can also directly check for recognition.

# In[20]:


#%%top
expr_parser = Parser(EXPR_GRAMMAR, start_symbol=EXPR_START, canonical=True)


# In[21]:


get_ipython().run_line_magic('top', 'assert expr_parser.can_parse(expr_input)')


# In[22]:


#%%top
bexpr_parser = Parser(BEXPR_GRAMMAR, canonical=True, start_symbol=BEXPR_START)


# In[23]:


get_ipython().run_line_magic('top', 'assert bexpr_parser.can_parse(bexpr_input)')


# #### Display
# 
# While converting to strings are easy, it is unsatisfying. We want to make our output look pretty, and inspect the tree structure of the parsed tree. So we define graphical tree display (code from fuzzingbook)

# In[24]:


from graphviz import Digraph


# In[25]:


from IPython.display import display, Image


# In[26]:


def zoom(v, zoom=True):
    # return v directly if you do not want to zoom out.
    if zoom:
        return Image(v.render(format='png'))
    return v


# In[27]:


class DisplayTree():
    def __init__(self):
        pass

    def extract_node(self, node, id):
        symbol, children, *annotation = node
        return symbol, children, ''.join(str(a) for a in annotation)
    
    def node_attr(self, dot, nid, symbol, ann):
        dot.node(repr(nid), symbol + ' ')
        
    def edge_attr(self, dot, start_node, stop_node):
        dot.edge(repr(start_node), repr(stop_node))
        
    def graph_attr(self, dot):
        dot.attr('node', shape='plain')
        
    def display(self, derivation_tree):
        counter = 0
        def traverse_tree(dot, tree, id=0):
            (symbol, children, annotation) = self.extract_node(tree, id)
            self.node_attr(dot, id, symbol, annotation)
            if children:
                for child in children:
                    nonlocal counter
                    counter += 1
                    child_id = counter
                    self.edge_attr(dot, id, child_id)
                    traverse_tree(dot, child, child_id)
        dot = Digraph(comment="Derivation Tree")
        self.graph_attr(dot)
        traverse_tree(dot, derivation_tree)
        return dot
    
    def __call__(self, dt):
        return self.display(dt)


# In[28]:


display_tree = DisplayTree()


# We are now ready to display the tree structure.

# In[29]:


get_ipython().run_line_magic('top', 'zoom(display_tree(expr_tree))')


# #### A Fuzzer

# In order to define abstraction, we need to be able to generate values based on a grammar. Our fuzzer is able to do that.

# In[30]:


import random


# In[31]:


random.seed(0)


# ##### The interface

# In[32]:


class Fuzzer:
    def __init__(self, grammar):
        self.grammar = grammar

    def fuzz(self, key='<start>', max_num=None, max_depth=None):
        raise NotImplemented()


# ##### The implementation

# The fuzzer tries to randomly choose an expansion when more than one expansion is available. If however, it goes beyond max_depth, then it chooses the cheapest nodes. The cheapest nodes are those nodes with minimum further expansion (no recursion).

# In[33]:


class LimitFuzzer(Fuzzer):
    def symbol_cost(self, grammar, symbol, seen):
        if symbol in self.key_cost: return self.key_cost[symbol]
        if symbol in seen:
            self.key_cost[symbol] = float('inf')
            return float('inf')
        v = min((self.expansion_cost(grammar, rule, seen | {symbol})
                    for rule in grammar.get(symbol, [])), default=0)
        self.key_cost[symbol] = v
        return v

    def expansion_cost(self, grammar, tokens, seen):
        return max((self.symbol_cost(grammar, token, seen)
                    for token in tokens if token in grammar), default=0) + 1

    def nonterminals(self, rule):
        return [t for t in rule if self.is_nt(t)]

    def iter_gen_key(self, key, max_depth):
        def get_def(t):
            if self.is_nt(t):
                return [t, None]
            else:
                return [t, []]

        cheap_grammar = {}
        for k in self.cost:
            # should we minimize it here? We simply avoid infinities
            rules = self.grammar[k]
            min_cost = min([self.cost[k][str(r)] for r in rules])
            #grammar[k] = [r for r in grammar[k] if self.cost[k][str(r)] == float('inf')]
            cheap_grammar[k] = [r for r in self.grammar[k] if self.cost[k][str(r)] == min_cost]

        root = [key, None]
        queue = [(0, root)]
        while queue:
            # get one item to expand from the queue
            (depth, item), *queue = queue
            key = item[0]
            if item[1] is not None: continue
            grammar = self.grammar if depth < max_depth else cheap_grammar
            chosen_rule = random.choice(grammar[key])
            expansion = [get_def(t) for t in chosen_rule]
            item[1] = expansion
            for t in expansion: queue.append((depth+1, t))
            #print("Fuzz: %s" % key, len(queue), file=sys.stderr)
        #print(file=sys.stderr)
        return root

    def gen_key(self, key, depth, max_depth):
        if key not in self.grammar:
            return (key, [])
        if depth > max_depth:
            #return self.gen_key_cheap_iter(key)
            clst = sorted([(self.cost[key][str(rule)], rule) for rule in self.grammar[key]])
            rules = [r for c,r in clst if c == clst[0][0]]
        else:
            rules = self.grammar[key]
        return (key, self.gen_rule(random.choice(rules), depth+1, max_depth))

    def gen_rule(self, rule, depth, max_depth):
        return [self.gen_key(token, depth, max_depth) for token in rule]

    def fuzz(self, key='<start>', max_depth=10):
        self._s = self.iter_gen_key(key=key, max_depth=max_depth)
        return self.tree_to_str(self._s)
   
    def is_nt(self, name):
        return (name[0], name[-1]) == ('<', '>')
 
    def tree_to_str(self, tree):
        name, children = tree
        if not self.is_nt(name): return name
        return ''.join([tree_to_str(c) for c in children])

    def tree_to_str(self, tree):
        expanded = []
        to_expand = [tree]
        while to_expand:
            (key, children, *rest), *to_expand = to_expand
            if is_nt(key):
                #assert children # not necessary
                to_expand = children + to_expand
            else:
                assert not children
                expanded.append(key)
        return ''.join(expanded)


    def __init__(self, grammar):
        super().__init__(grammar)
        self.key_cost = {}
        self.cost = self.compute_cost(grammar)

    def compute_cost(self, grammar):
        cost = {}
        for k in grammar:
            cost[k] = {}
            for rule in grammar[k]:
                cost[k][str(rule)] = self.expansion_cost(grammar, rule, set())
            if len(grammar[k]):
                assert len([v for v in cost[k] if v != float('inf')]) > 0
        return cost


# In[34]:


#%%top
expr_fuzzer = LimitFuzzer(EXPR_GRAMMAR)
bexpr_fuzzer = LimitFuzzer(BEXPR_GRAMMAR)


# In[35]:


get_ipython().run_line_magic('top', 'expr_fuzzer.fuzz(EXPR_START)')


# In[36]:


get_ipython().run_line_magic('top', 'bexpr_fuzzer.fuzz(BEXPR_START)')


# ### Library functions

# Copying a grammar

# In[37]:


def copy_grammar(g):
    return {k:[[t for t in r] for r in g[k]] for k in g}


# Produce a sorted list of rules

# In[38]:


def sort_rules(rules):
    return sorted([list(r) for r in rules])


# #### Nullability

# In[39]:


EPSILON=''
def get_rules(g): return [(k, e) for k, a in g.items() for e in a]
def get_terminals(g):
    return set(t for k, expr in get_rules(g) for t in expr if t not in g)
def fixpoint(f):
    def helper(*args):
        while True:
            sargs = repr(args)
            args_ = f(*args)
            if repr(args_) == sargs:
                return args
            args = args_
    return helper


@fixpoint
def nullable_(grules, e):
    for A, expression in grules:
        if all((token in e)  for token in expression): e |= {A}
    return (grules, e)

def nullable_nt(grammar):
    return nullable_(get_rules(grammar), set())[1]

def is_nullable(grammar, start):
    return start in nullable_nt(grammar)


# In[40]:


get_ipython().run_line_magic('top', "nullable_nt({'<a>': [['a'], ['<b>']], '<b>': [[]], '<c>': [['c']]})")


# #### Check for empty grammar
# 
# One can check for empty grammar by first removing all terminals and check the resulting grammar for nullability.

# In[41]:


def remove_all_terminals(grammar):
    return {k:[[t for t in r if is_nt(t)] for r in grammar[k]] for k in grammar} 


# In[42]:


get_ipython().run_line_magic('top', '_expr_grammar_tmp1 = remove_all_terminals(EXPR_GRAMMAR)')


# In[43]:


get_ipython().run_line_magic('top', 'nullable_nt(_expr_grammar_tmp1)')


# In[44]:


get_ipython().run_line_magic('top', 'nullable_nt(EXPR_GRAMMAR)')


# In[45]:


empty_g, empty_s = {'<start>': [['<expr>']], '<expr>': [['<expr>']]}, '<start>'


# In[46]:


get_ipython().run_line_magic('top', 'nullable_nt(empty_g)')


# In[47]:


get_ipython().run_line_magic('top', 'is_nullable(empty_g, empty_s)')


# In[48]:


get_ipython().run_line_magic('top', 'is_nullable(EXPR_GRAMMAR, EXPR_START)')


# In[49]:


def is_cfg_empty(grammar, start):
    # first remove all terminals
    null_g = remove_all_terminals(grammar)
    # then check if start is nullable.
    return not is_nullable(null_g, start)


# In[50]:


get_ipython().run_line_magic('top', 'is_cfg_empty(EXPR_GRAMMAR, EXPR_START)')


# In[51]:


get_ipython().run_line_magic('top', 'is_cfg_empty(empty_g, empty_s)')


# #### Show

# In[52]:


def show_grammar(grammar, verbose=0):
    r = 0
    k = 0
    for key in grammar:
        k += 1
        if verbose > -1: print(key,'::=')
        for rule in grammar[key]:
            r += 1
            if verbose > 1:
                pre = r
            else:
                pre = ''
            if verbose > -1:
                print('%s|   ' % pre, ' '.join([t if is_nt(t) else repr(t) for t in rule]))
        if verbose > 0:
            print(k, r)
    print(k, r)


# In[53]:


Gs = show_grammar


# In[54]:


get_ipython().run_line_magic('top', 'Gs(EXPR_GRAMMAR)')


# #### Split token

# In[55]:


def tsplit(token):
    assert token[0], token[-1] == ('<', '>')
    front, *back = token[1:-1].split(None, 1)
    return front, ' '.join(back)


# #### The stem
# 
# The stem of a token is the original nonterminal it corresponds to, before refinement.

# In[56]:


def stem(token):
    return tsplit(token)[0].strip()


# #### Normalization

# In[57]:


def normalize(key):
    return '<%s>' % stem(key)


# In[58]:


def normalize_grammar(g):
    return {normalize(k):list({tuple([normalize(t) if is_nt(t) else t for t in r]) for r in g[k]}) for k in g}


# #### The refinement
# 
# The refinement of a token is the part after the initial space in a token which corresponds to how the refined token came to be (based on adding/negating other faults.)

# In[59]:


def refinement(token):
    return tsplit(token)[1]


# In[60]:


def is_refined_key(key):
    assert is_nt(key)
    return (' ' in key)


# In[61]:


def is_base_key(key):
    return not is_refined_key(key)


# In[62]:


get_ipython().run_line_magic('top', "assert refinement('<a b>') == 'b'")


# In[63]:


get_ipython().run_line_magic('top', "assert is_refined_key('<a b>')")


# In[64]:


get_ipython().run_line_magic('top', "assert is_base_key('<a>')")


# In[65]:


get_ipython().run_line_magic('top', "assert not is_base_key('<a b>')")


# In[66]:


def is_refined_rule(rule):
    for k in rule:
        if is_nt(k) and is_refined_key(k): return True
    return False


# #### Find Node
# Finding nodes given the path

# In[67]:


def find_node(node, path):
    name, children, *rest = node
    if not path:
        return node
    p, *path = path
    for i,c in enumerate(children):
        if i == p:
            return find_node(c, path)
    return None


# In[68]:


get_ipython().run_line_magic('top', "assert find_node(expr_tree, [0,1]) == (' + ', [])")


# In[69]:


get_ipython().run_line_magic('top', "assert find_node(expr_tree, [0,2,0,0,2]) == (')', [])")


# In[70]:


def Ns(expr, paths):
    for path in paths:
        n = find_node(expr, path)
        print(n[0], tree_to_str(n))


# In[71]:


get_ipython().run_cell_magic('top', '', 'Ns(expr_tree, [[0,2]])\nNs(expr_tree, [[0,2,0]])\nNs(expr_tree, [[0,2,0,0]])\nNs(expr_tree, [[0,2,0,0,1]])')


# #### Path to Key
# Get the path to where key is defined. This is not for parse trees where the same nonterminal may be found in multiple nodes, but for trees where the key at each node is unique.

# In[72]:


def path_to_key(tree, key):
    if tree is None: return None
    name, children = tree
    if key == name:
        return []
    for i,c in enumerate(children):
        p = path_to_key(c, key)
        if p is not None:
            return [i] + p
    return None


# #### Rule to normalized rule

# In[73]:


def rule_to_normalized_rule(rule):
    return [normalize(t) if is_nt(t) else t for t in rule]


# In[74]:


def normalized_rule_match(r1, r2):
    return rule_to_normalized_rule(r1) == rule_to_normalized_rule(r2)


# In[75]:


def rule_normalized_difference(rulesA, rulesB):
    rem_rulesA = rulesA
    for ruleB in rulesB:
        rem_rulesA = [rA for rA in rem_rulesA if not normalized_rule_match(rA, ruleB)]
    return rem_rulesA


# In[76]:


def rules_normalized_match_to_rule(rulesA, rB):
    selA = [rA for rA in rulesA if normalized_rule_match(rA, rB)]
    return selA


# In[77]:


get_ipython().run_line_magic('top', "EXPR_GRAMMAR['<factor>']")


# In[78]:


get_ipython().run_cell_magic('top', '', "assert rule_normalized_difference(EXPR_GRAMMAR['<factor>'], [['(', '<expr F1>', ')']]) == [\n    ['+', '<factor>'],\n    ['-', '<factor>'],\n    ['<integer>', '.', '<integer>'],\n    ['<integer>']]")


# In[79]:


get_ipython().run_cell_magic('top', '', "assert rule_normalized_difference(EXPR_GRAMMAR['<factor>'], [['(', '<expr>', ')'], ['<integer>']]) == [\n    ['+', '<factor>'],\n    ['-', '<factor>'],\n    ['<integer>', '.', '<integer>']]")


# #### Node to normalized rule

# In[80]:


def node_to_rule(node):
    name, children, *rest = node
    return [c[0] for c in children]


# In[81]:


get_ipython().run_line_magic('top', 'node_to_rule(find_node(expr_tree, [0]))')


# In[82]:


get_ipython().run_line_magic('top', 'n = find_node(expr_tree, [0,2,0, 0]); n')


# In[83]:


get_ipython().run_line_magic('top', 'node_to_rule(n)')


# In[84]:


def node_to_normalized_rule(node):
    return rule_to_normalized_rule(node_to_rule(node))


# In[85]:


get_ipython().run_line_magic('top', 'node_to_normalized_rule(n)')


# #### Replace Tree

# In[86]:


def replace_tree(node, path, newnode):
    if not path:
        return newnode
    name, children = node
    hd, *subpath = path
    assert hd < len(children)
    new_children = []
    for i,c in enumerate(children):
        if i == hd:
            c_ = replace_tree(c, subpath, newnode)
        else:
            c_ = c
        new_children.append(c_)
    return (name, new_children)


# In[87]:


get_ipython().run_line_magic('top', "assert tree_to_str(find_node(expr_tree, [0,2,0,0])) == '((2 * 3 / 4))'")


# In[88]:


get_ipython().run_line_magic('top', "assert tree_to_str(replace_tree(expr_tree, [0, 2, 0, 0], ('1', []))) == '1 + 1'")


# #### Validate Tree
# 
# Validating a parse tree

# In[89]:


def validate_tree(tree, grammar):
    def keys(arr):
        return [a[0] for a in arr]
    name, children, *rest = tree
    if not is_nt(name): return True
    
    seen = False
    for rule in grammar[name]:
        if keys(children) == rule:
            seen = True
    assert seen, name + ' needs ' + repr(grammar[name])
    for c in children:
        validate_tree(c, grammar)


# In[90]:


get_ipython().run_line_magic('top', 'validate_tree(expr_tree, EXPR_GRAMMAR)')


# In[91]:


get_ipython().run_cell_magic('top', '', "try:\n    validate_tree(replace_tree(expr_tree, [0, 2, 0, 0], ('1', [])), EXPR_GRAMMAR)\nexcept AssertionError as e:\n    print(e)")


# In[92]:


get_ipython().run_cell_magic('top', '', "rt = replace_tree(expr_tree, [0, 2, 0, 0], ('<factor>', [('<integer>', [('<digit>',[('1', [])])])]))\nvalidate_tree(rt, EXPR_GRAMMAR)")


# In[93]:


get_ipython().run_line_magic('top', "assert tree_to_str(rt) == '1 + 1'")


# #### Remove empty keys

# Remove keys that do not have a definition.

# In[94]:


def find_empty_keys(g):
    return [k for k in g if not g[k]]


# In[95]:


def remove_key(k, g):
    new_g = {}
    for k_ in g:
        if k_ == k:
            continue
        else:
            new_rules = []
            for rule in g[k_]:
                new_rule = []
                for t in rule:
                    if t == k:
                        # skip this rule
                        new_rule = None
                        break
                    else:
                        new_rule.append(t)
                if new_rule is not None:
                    new_rules.append(new_rule)
            new_g[k_] = new_rules
    return new_g


# In[96]:


def remove_empty_keys(g):
    new_g = copy_grammar(g)
    removed_keys = []
    empty_keys = find_empty_keys(new_g)
    while empty_keys:
        for k in empty_keys:
            removed_keys.append(k)
            new_g = remove_key(k, new_g)
        empty_keys = find_empty_keys(new_g)
    return new_g, removed_keys


# In[97]:


tmp_g_empty_ = {
    '<start>': [['<v>']],
    '<v>': [
        ['<expr>'],
        ['<digit>']
    ],
    '<expr>': [['<digit>'],
              ['x', '<empty>']],
    '<digit>': [['0']],
    '<empty>': []
}


# In[98]:


get_ipython().run_cell_magic('top', '', 'tmp_g, removed_keys = remove_empty_keys(tmp_g_empty_)\nGs(tmp_g)\nremoved_keys')


# #### Find all nonterminals

# In[99]:


def find_all_nonterminals(g):
    lst = []
    for k in g:
        for r in g[k]:
            for t in r:
                if is_nt(t):
                    lst.append(t)
    return list(sorted(set(lst)))


# In[100]:


get_ipython().run_line_magic('top', 'find_all_nonterminals(tmp_g)')


# #### Undefined keys

# In[101]:


def undefined_keys(grammar):
    keys = find_all_nonterminals(grammar)
    return [k for k in keys if k not in grammar]


# ####  Remove unused keys

# Removes unused nonterminal keys

# In[102]:


def remove_unused_keys(grammar, start_symbol):
    def strip_key(grammar, key, order):
        rules = sort_rules(grammar[key])
        old_len = len(order)
        for rule in rules:
            for token in rule:
                if is_nt(token):
                    if token not in order:
                        order.append(token)
        new = order[old_len:]
        for ckey in new:
            strip_key(grammar, ckey, order)
    if start_symbol not in grammar:
        return {}, []

    order = [start_symbol]
    strip_key(grammar, start_symbol, order)
    if len(order) != len(grammar.keys()):
        stripped = [k for k in grammar if k not in order]
        #if stripped:
        #    print("Stripping: %s" % str(stripped))
        faulty = [k for k in order if k not in grammar]
        assert not faulty
    new_g = {k: [list(r) for r in sort_rules(grammar[k])] for k in order}
    return new_g, [k for k in grammar if k not in new_g]


# In[103]:


get_ipython().run_cell_magic('top', '', "tmp_g_unused_ = copy_grammar(EXPR_GRAMMAR)\ntmp_g_unused_['<unused>'] = [['1']]\nGs(tmp_g_unused_,-1)\ntmp_g, removed_keys = remove_unused_keys(tmp_g_unused_, EXPR_START)\nGs(tmp_g)\nremoved_keys")


# #### Validate grammar

# In[104]:


def validate_grammar(grammar, start_symbol, log=False):
    assert len(grammar[start_symbol]) == 1 # otherwise parser will not work
    for r in grammar[start_symbol]:
        assert len([t for t in r if is_nt(t)]) == 1 # otherwise disjunction will not work
    def strip_key(grammar, key, order):
        rules = sort_rules(grammar[key])
        old_len = len(order)
        for rule in rules:
            for token in rule:
                if is_nt(token):
                    if token not in order:
                        order.append(token)
        new = order[old_len:]
        for ckey in new:
            strip_key(grammar, ckey, order)
    if start_symbol not in grammar:
        return {}, []

    order = [start_symbol]
    strip_key(grammar, start_symbol, order)
    valid = True
    if len(order) != len(grammar.keys()):
        unused = [k for k in grammar if k not in order]
        faulty = [k for k in order if k not in grammar]
        if faulty or unused:
            if log:
                print('faulty:', faulty)
                print('unused:', unused)
            return False
    for (k1,k2) in zip(sorted(order), sorted(grammar.keys())):
        if k1 != k2:
            valid = False
            if log:
                print('order:', k1, 'grammar:', k2)
        else:
            if log:
                print(k1, k2)
    for k in grammar:
        if not grammar[k]:
            return False
    return valid


# In[105]:


get_ipython().run_line_magic('top', 'assert validate_grammar(EXPR_GRAMMAR, EXPR_START, True)')


# In[106]:


tmp_g_empty_


# In[107]:


get_ipython().run_line_magic('top', "assert not validate_grammar(tmp_g_empty_, '<start>', True)")


# In[108]:


get_ipython().run_line_magic('top', "assert not validate_grammar(tmp_g_unused_, '<start>', True)")


# #### Finding reachable keys

# In[109]:


def find_reachable_keys(grammar, key, reachable_keys=None, found_so_far=None):
    if reachable_keys is None: reachable_keys = {}
    if found_so_far is None: found_so_far = set()

    for rule in grammar[key]:
        for token in rule:
            if not is_nt(token): continue
            if token in found_so_far: continue
            found_so_far.add(token)
            if token in reachable_keys:
                for k in reachable_keys[token]:
                    found_so_far.add(k)
            else:
                keys = find_reachable_keys(grammar, token, reachable_keys, found_so_far)
                # reachable_keys[token] = keys <- found_so_far contains results from earlier
    return found_so_far


# In[110]:


get_ipython().run_cell_magic('top', '', 'for key in EXPR_GRAMMAR:\n    keys = find_reachable_keys(EXPR_GRAMMAR, key, {})\n    print(key, keys)')


# Finding recursive keys

# In[111]:


def reachable_dict(grammar):
    reachable = {}
    for key in grammar:
        keys = find_reachable_keys(grammar, key, reachable)
        reachable[key] = keys
    return reachable


# In[112]:


get_ipython().run_cell_magic('top', '', 'reachable_dict(EXPR_GRAMMAR)')


# #### Simplify boolean expressions

# In[113]:


import sympy


# In[114]:


class OrB:
    def __init__(self, a, b): 
        if b is None:
            assert isinstance(a, list)
            if a[1:]:
                self.a, self.b = a[0], OrB(a[1:], None)
            else:
                self.a, self.b = a[0], None 
        else:       
            self.a, self.b = a, b
    def __str__(self):
        if self.b is None: return str(self.a)
        return 'or(%s,%s)' % (str(self.a), str(self.b))
class AndB:
    def __init__(self, a, b):
        if b is None:
            assert isinstance(a, list)
            if a[1:]:
                self.a, self.b = a[0], AndB(a[1:], None)
            else:
                self.a, self.b = a[0], None 
        else:
            self.a, self.b = a, b
    def __str__(self):
        if self.b is None: return str(self.a)
        return 'and(%s,%s)' % (str(self.a), str(self.b))
class NegB:
    def __init__(self, a): self.a = a
    def __str__(self): return 'neg(%s)' % str(self.a)
class B:
    def __init__(self, a): self.a = a
    def __str__(self): return str(self.a)


# In[115]:


def convert_to_sympy(bexpr, symargs=None):
    def get_op(node):
        assert node[0] == '<bop>'
        return node[1][0][0]
    if symargs is None:
        symargs = {}
    name, children = bexpr
    assert name == '<bexpr>'
    if len(children) == 1: # fault node
        name = tree_to_str(children[0])
        if not name: return None, symargs
        if name not in symargs:
            symargs[name] = sympy.symbols(name)
        return symargs[name], symargs
    else:
        operator = get_op(children[0])
        if operator == 'and':
            a,_ = convert_to_sympy(children[2], symargs)
            comma = children[3]
            assert comma[0] == ',', comma[0]
            b,_ = convert_to_sympy(children[4], symargs)
            if a is None:
                assert b is not None
                return b, symargs
            elif b is None:
                assert a is not None
                return a, symargs
            return sympy.And(a, b), symargs
        elif operator == 'or':
            a,_ = convert_to_sympy(children[2], symargs)
            comma = children[3]
            assert comma[0] == ',', comma[0]
            b,_ = convert_to_sympy(children[4], symargs)
            if a is None:
                assert b is not None
                return b, symargs
            elif b is None:
                assert a is not None
                return a, symargs
            return sympy.Or(a, b), symargs
        elif operator == 'neg':
            a,_ = convert_to_sympy(children[2], symargs)
            return sympy.Not(a), symargs
        else:
            assert False


# In[116]:


def convert_sympy_to_bexpr(sexpr, log=False):
    if isinstance(sexpr, sympy.Symbol):
        return B(str(sexpr))
    elif isinstance(sexpr, sympy.Not):
        return NegB(convert_sympy_to_bexpr(sexpr.args[0]))
    elif isinstance(sexpr, sympy.And):
        sym_vars = sorted([convert_sympy_to_bexpr(a) for a in sexpr.args], key=str)
        assert sym_vars
        return AndB(sym_vars, None)
    elif isinstance(sexpr, sympy.Or):
        sym_vars = sorted([convert_sympy_to_bexpr(a) for a in sexpr.args], key=str)
        assert sym_vars
        return OrB(sym_vars, None)
    else:
        if log: print(repr(sexpr))
        assert False


# In[117]:


def bexpr_parse(k):
    bexpr_parser = Parser(BEXPR_GRAMMAR, canonical=True, start_symbol=BEXPR_START)
    bparse_tree = list(bexpr_parser.parse(k))[0]
    bexpr = bparse_tree[1][0]
    return bexpr


# In[118]:


def simplify_bexpr(bexpr_input):
    bexpr_tree = bexpr_parse(bexpr_input)
    e0, defs = convert_to_sympy(bexpr_tree)
    e1 = sympy.to_dnf(e0)
    e2 = convert_sympy_to_bexpr(e1)
    v = str(e2)
    my_keys = [k for k in defs]
    for k in my_keys:
        del defs[k]
    return v


# In[119]:


get_ipython().run_cell_magic('top', '', "bexpr_input = 'and(and(and(and(and(neg(+F2),neg(L2_2)),neg(+F2)),and(neg(+F2),neg(L2_2))),and(and(neg(+F2),neg(L2_2)),neg(+F2))),and(and(and(neg(+F2),neg(L2_2)),neg(+F2)),and(neg(+F2),neg(L2_2))))'\nsimplified_bexpr_input = simplify_bexpr(bexpr_input)")


# In[120]:


get_ipython().run_line_magic('top', 'simplified_bexpr_input')


# In[121]:


def show_simplified_grammar(grammar, no_dups=True, verbose=0):
    def sb(key):
        if is_base_key(key):
            return key
        return '<%s %s>' %(stem(key), simplify_bexpr(refinement(key)))
    r = 0
    k = 0
    seen = {}
    for key in grammar:
        k += 1
        sk = sb(key)
        if sk in seen:
            continue
        else:
            seen[sk] = True
        if verbose > -1: print(sk,'::=')
        for rule in grammar[key]:
            r += 1
            if verbose > 1:
                pre = r
            else:
                pre = ''
            if verbose > -1:
                print('%s|   ' % pre, ' '.join([sb(t) if is_nt(t) else repr(t) for t in rule]))
        if verbose > 0:
            print(k, r)
    print(k, r)


# In[122]:


Gs_s = show_simplified_grammar


# #### Remove shadowed refined rules

# ##### is_A_more_refined_than_B
# 
# Compare the positions of `ruleA` and `ruleB` in `porder` of base `grammar`.
# In the porder, we assume the most general ones will come first.

# In[123]:


def is_A_more_refined_than_B(ruleA, ruleB, porder):
    if len(ruleA) != len(ruleB): return False
    for a_, b_ in zip(ruleA, ruleB):
        if not is_nt(a_) or not is_nt(b_):
            if a_ != b_: return False
            continue
        a = normalize(a_) 
        b = normalize(b_)
        if a != b: return False
        if a not in porder or b not in porder: return None
        pkA = path_to_key(porder[a], a_)
        pkB = path_to_key(porder[b], b_)
        if pkA is None or pkB is None: return None # we dont know.
        pA = ' '.join([str(s) for s in pkA])
        pB = ' '.join([str(s) for s in pkB])
        # more general should be included in the more specific
        if pB not in pA: return False
    return True


# In[124]:


get_ipython().run_cell_magic('top', '', "my_porder = {'<a>' : ('<a>',[('<a 1>', [('<a 2>',[('<a 3>',[])])])]),\n             '<b>':  ('<b>',[('<b 1>', [('<b 2>',[('<b 3>',[])])])]),\n             '<c>':  ('<c>',[('<c 1>', [('<c 2>',[('<c 3>',[])])])])}")


# In[125]:


get_ipython().run_line_magic('top', "assert not is_A_more_refined_than_B(['<a 1>', '<b 1>', '<c 1>'], ['<a 2>', '<b 2>', '<c 2>'], my_porder)")


# In[126]:


get_ipython().run_line_magic('top', "assert is_A_more_refined_than_B(['<a 3>', '<b 3>', '<c 3>'], ['<a 2>', '<b 2>', '<c 2>'], my_porder)")


# In[127]:


get_ipython().run_line_magic('top', "assert is_A_more_refined_than_B(['<a 2>', '<b 2>', '<c 3>'], ['<a 2>', '<b 2>', '<c 2>'], my_porder)")


# In[128]:


get_ipython().run_line_magic('top', "assert not is_A_more_refined_than_B(['<a 3>', '<b 3>', '<c 1>'], ['<a 2>', '<b 2>', '<c 2>'], my_porder)")


# ##### Get general rule

# In[129]:


def get_general_rule(ruleA, rules, porder):
    unknown = 0
    for r in rules:
        v = is_A_more_refined_than_B(ruleA, r, porder)
        if v is None:
            # we dont know about this.
            unknown += 1
            continue
        elif v:
            return r, unknown
    return None, unknown


# In[130]:


get_ipython().run_cell_magic('top', '', "get_general_rule(['<a 1>', '<b 1>', '<c 1>'], [\n    ['<a 1>', '<b 1>', '<c 1>'],\n    ['<a 2>', '<b 2>', '<c 2>'],\n    ['<a 3>', '<b 3>', '<c 3>'],\n], my_porder)")


# In[131]:


get_ipython().run_cell_magic('top', '', "get_general_rule(['<a 1>', '<b 1>', '<c 3>'], [\n    ['<a 1>', '<b 1>', '<c 1>'],\n    ['<a 2>', '<b 2>'],\n    ['<a 3>'],\n], my_porder)")


# In[132]:


get_ipython().run_cell_magic('top', '', "get_general_rule(['<a 1>', '<b 2>'], [\n    ['<a 1>', '<b 1>', '<c 1>'],\n    ['<a 2>', '<b 2>'],\n    ['<a 3>'],\n], my_porder)")


# ##### Is keyA more refined than keyB

# In[133]:


def is_keyA_more_refined_than_keyB(keyA, keyB, porder, grammar):
    # essential idea of comparing two keys is this:
    # One key is smaller than the other if for any given rule in the first, there exist another rule that is larger
    # than that in the second key.
    # a rule is smaller than another if all tokens in that rule is either equal (matching) or smaller than
    # the corresponding token in the other.
    
    # if normalize(keyB) == keyB: return True # normalized key is always the top (and may not exist in grammar)
    
    A_rules = grammar[keyA]
    B_rules = grammar[keyB]
   
    for A_rule in A_rules:
        v,unk = get_general_rule(A_rule, B_rules, porder)
        if v is None:
            if unk:
                return None # dont know
            return False
        # There is a more general rule than A_rule in B_rules
    return True


# ##### Insert into partial order

# In[134]:


def insert_into_porder(my_key, porder, grammar):
    def update_tree(my_key, tree, grammar):
        if tree is None: return True, (my_key, [])
        k, children = tree
        if is_base_key(my_key):
            if not is_base_key(k):
                return True, (my_key, [tree])
            else:
                return False, tree
 
        v = is_keyA_more_refined_than_keyB(my_key, k, porder, grammar)
        if is_base_key(k): v = True
        # if v is unknown...
        if v: # we should go into the children
            if not children:
                #print('>', 0)
                return True, (k, [(my_key, [])])
            new_children = []
            updated = False
            for c in children:
                u, c_ = update_tree(my_key, c, grammar)
                if u: updated = True
                new_children.append(c_)
            #print('>', 1)
            return updated, (k, new_children)
        else:
            #v = is_keyA_more_refined_than_keyB(k, my_key, porder, grammar)
            if v:
                #this should be the parent of tree
                #print('>', 2)
                return True, (my_key, [tree])
            else:
                # add as a sibling -- but only if we have evidence.
                if v is not None:
                    #print('>', 3)
                    return True, (k, children + [(my_key, [])])
                else:
                    return False, tree
    key = normalize(my_key)
    updated, v = update_tree(my_key, porder.get(key, None), grammar)
    if updated:
        porder[key] = v
    return updated


# ##### Is key in partial order

# In[135]:


def is_key_in_porder(key, tree):
    if tree is None: return False
    name, children = tree
    if name == key:
        return True
    for c in children:
        if is_key_in_porder(key, c):
            return True
    return False


# ##### Identify partial orders of nonterminals from a grammar

# In theory, identifying partial orders is simple once you have the machinary for `and` and `neg`. To find if a given nonterminal `A` is more refined than `B`, do `A - B` and `B - A` and check which of these are empty. Since we do not have the machinery yet, doing it here without that.

# In[136]:


def identify_partial_orders(grammar):
    porder = {}
    cont = True
    while cont:
        cont = False
        for k in grammar:
            nkey = normalize(k)
            if is_key_in_porder(k, porder.get(nkey, None)):
                continue
            updated = insert_into_porder(k, porder, grammar)
            if not updated:
                continue
            cont = True
    #for k in grammar:
    #    assert k in porder
    return porder


# In[137]:


get_ipython().run_cell_magic('top', '', 'po = identify_partial_orders(EXPR_GRAMMAR); po')


# In[138]:


def rule_is_redundant(rule, rules, porder):
    # a rule is redundant if there is another in the rules that is more general.
    grule, unknown = get_general_rule(rule, [r for r in rules if r != rule], porder)
    if grule:
        return True
    return False


# In[139]:


def remove_redundant_rules(grammar, porder=None):
    if porder is None:
        porder = identify_partial_orders(grammar)
    else:
        if porder == {}:
            _porder = identify_partial_orders(grammar)
            porder.update(_porder)
        else:
            pass
        
    new_g = {}
    removed_rules = 0
    for key in grammar:
        ruleset = list({tuple(r) for r in grammar[key]})
        cont = True
        while cont:
            cont = False
            for rule in ruleset:
                if rule_is_redundant(rule, ruleset, porder):
                    ruleset = [r for r in ruleset if r != rule]
                    removed_rules += 1
                    cont = True
                else:
                    continue
            new_g[key] = ruleset
    return new_g, removed_rules


# In[140]:


get_ipython().run_cell_magic('top', '', 'g, n = remove_redundant_rules(EXPR_GRAMMAR);\nGs(g)')


# #### Grammar GC

# We will redefine this later when the negation comes in.

# In[141]:


def grammar_gc(grammar, start_symbol, options=(1,2), log=False):
    g = grammar
    while True:
        if 1 in options:
            g0, empty_keys = remove_empty_keys(g)
        else:
            g0, empty_keys = g, []
        for k in g0:
            for rule in g0[k]:
                for t in rule: assert type(t) is str

        if 2 in options:
            g1, unused_keys = remove_unused_keys(g0, start_symbol)
        else:
            g1, unused_keys = g0, []
        for k in g1:
            for rule in g1[k]:
                for t in rule: assert type(t) is str
        g = g1
        if log:
            print('GC: ', unused_keys, empty_keys)
        if not (len(unused_keys) + len(empty_keys)):
            break
 
    # removing redundant rules is slightly dangerous. It can remove things
    # like finite limit of a recursion because the limit is more refined than
    # other rules.
    # E.g
    #  <E f> = <E f>
    #        | <E l>
    #  <E l> = 1
    # Here, <E f> is more general than <E l> precisely because <E l> exists
    # but our partial orders are not intelligent enough (todo: check)
    if 3 in options:
        g2, redundant_rules = remove_redundant_rules(g)
    else:
        g2, redundant_rules = g, 0
        
    #if 4 in options:
    #  We need to incorporate simplify booleans
    #else:
    return g2, start_symbol


# ## Inserting a fault

# The output that we get from `ddset` has nodes marked. So, we define a way to mark nodes as abstract.

# ### Mark the abstract nodes
# 
# Given a path, we mark the node as abstract.

# In[142]:


def mark_path_abstract(tree, path):
    name, children = find_node(tree, path)
    new_tree = replace_tree(tree, path, (name, children, {'abstract': True}))
    return new_tree


# First, we locate a suitable node.

# In[143]:


get_ipython().run_cell_magic('top', '', "abs_path_1 = [0,2,0,0,1,0,0,1]\nassert tree_to_str(find_node(expr_tree, abs_path_1)) == '2 * 3 / 4'")


# In[144]:


get_ipython().run_cell_magic('top', '', 'v = mark_path_abstract(expr_tree, abs_path_1); v')


# Given a tree with some nodes marked abstract, go through the tree, and mark everything else as concrete. Default is to mark a node as concrete.

# In[145]:


def mark_concrete_r(tree):
    name, children, *abstract_a = tree
    abstract = {'abstract': False} if not abstract_a else abstract_a[0]
    return (name, [mark_concrete_r(c) for c in children], abstract)


# In[146]:


get_ipython().run_cell_magic('top', '', 't = mark_concrete_r(v); t')


# A way to display the abstracted tree

# In[147]:


def till_abstract(node):
    name, children, *rest = node
    if rest[-1]['abstract']:
        return (name + '*', [])
    return (name, [till_abstract(c) for c in children], *rest)


# In[148]:


get_ipython().run_cell_magic('top', '', 'zoom(display_tree(till_abstract(t)))')


# In[149]:


def Da(t):
    return zoom(display_tree(till_abstract(t)))


# In[150]:


#%%top
abs_t1_ = find_node(expr_tree, [0, 2])
tree_to_str(abs_t1_), abs_t1_[0]


# In[151]:


#%%top
abs_t1 = ('<start>', [abs_t1_])


# In[152]:


get_ipython().run_line_magic('top', 'validate_tree(abs_t1, EXPR_GRAMMAR)')


# In[153]:


#%%top
t_abs_p1 = [0, 0, 0, 1, 0, 0, 1]
Ts(find_node(abs_t1, t_abs_p1))


# We now define a function to check if a given node is abstract or not.

# In[154]:


def is_node_abstract(node):
    name, children, *abstract_a = node
    if not abstract_a:
        return True
    else:
        return abstract_a[0]['abstract']


# In[155]:


def tree_to_str_a(tree):
    name, children, *general_ = tree
    if not is_nt(name): return name
    if is_node_abstract(tree):
        return name
    return ''.join([tree_to_str_a(c) for c in children])


# In[156]:


Ta = tree_to_str_a


# In[157]:


get_ipython().run_line_magic('top', 'tree_to_str_a(t)')


# In[158]:


#%%top
abs_tree1 = mark_concrete_r(mark_path_abstract(abs_t1, t_abs_p1)); abs_tree1


# In[159]:


get_ipython().run_line_magic('top', 'Ta(abs_tree1)')


# In[160]:


get_ipython().run_line_magic('top', 'Da(abs_tree1)')


# In[161]:


def mark_abstract_nodes(tree, paths):
    for path in paths:
        tree = mark_path_abstract(tree, path)
    return mark_concrete_r(tree)


# In[162]:


get_ipython().run_line_magic('top', 'Ta(mark_abstract_nodes(abs_t1, []))')


# ### An abstract pattern from DDSET - F1

# In[163]:


get_ipython().run_line_magic('top', 'Ta(mark_abstract_nodes(abs_t1, [t_abs_p1]))')


# **IMPORTANT** This is what we expect our input to be. That is, a full parse tree of a minimized string with abstractions (typed holes) indicated.

# ### Finding characterizing node
# 
# A characterizing node is the lowest node that completely contains the given pattern.

# In[164]:


def find_charecterizing_node(tree):
   name, children, gen = tree
   if len(children) == 1:
       return find_charecterizing_node(children[0])
   return tree


# In[165]:


#%%top
abs_tree_cnode1 = find_charecterizing_node(abs_tree1); abs_tree_cnode1


# As can be seen, the `<factor>` node completely contains the fault pattern.

# In[166]:


get_ipython().run_line_magic('top', 'abs_tree1[0], Ts(abs_tree1)')


# In[167]:


get_ipython().run_line_magic('top', 'abs_tree_cnode1[0], Ts(abs_tree_cnode1)')


# Now, we want to add our grammar the keys that are required to cause a failure. For that, we first extract the local grammar that reproduces the fault pattern 

# ### Pattern Grammar

# In[168]:


get_ipython().run_line_magic('top', 'abs_tree_cnode1')


# In[169]:


def mark_faulty_name(symbol, prefix, v):
   return '<%s L%s_%s>'% (symbol[1:-1], prefix, v)


# In[170]:


def mark_faulty_nodes(node, prefix, counter=None):
    if counter is None: counter = {}
    symbol, children, *abstract = node
    if is_node_abstract(node): # we dont markup further
        return node
    if symbol not in counter: counter[symbol] = 0
    counter[symbol] += 1
    v = str(counter[symbol])
    if is_nt(symbol):
        return (mark_faulty_name(symbol, prefix, v),
                [mark_faulty_nodes(c, prefix, counter) for c in children],
                *abstract)
    else:
        assert not children
        return (symbol, children, *abstract)


# In[171]:


get_ipython().run_line_magic('top', "display_tree(mark_faulty_nodes(abs_tree_cnode1, '1'))")


# In[172]:


get_ipython().run_line_magic('top', "c_node1 = mark_faulty_nodes(abs_tree_cnode1, '1')")


# In[173]:


get_ipython().run_line_magic('top', 'f_node1 = c_node1')


# In[174]:


def faulty_node_to_grammar(tree, grammar=None):
    if grammar is None: grammar = {}
    if is_node_abstract(tree): return grammar
    name, children, *rest = tree
    tokens = []
    if name not in grammar: grammar[name] = []
    for c in children:
        n, cs, *rest = c
        tokens.append(n)
        if is_nt(n):
            faulty_node_to_grammar(c, grammar)
    grammar[name].append(tuple(tokens))
    return grammar, tree[0]


# In[175]:


get_ipython().run_cell_magic('top', '', 'g, s = faulty_node_to_grammar(abs_tree_cnode1)\nGs(g)\ns')


# In[176]:


def faulty_node_to_pattern_grammar(tree, prefix, grammar=None):
    ltree = mark_faulty_nodes(tree, prefix)
    return faulty_node_to_grammar(ltree)


# In[177]:


#%%top
lg1, ls1 = faulty_node_to_pattern_grammar(abs_tree_cnode1, '1')


# In[178]:


get_ipython().run_cell_magic('top', '', 'Gs(lg1)\nls1')


# In[179]:


get_ipython().run_line_magic('top', 'dd_tree_abs1 = abs_tree1')


# In[180]:


#%top
node_faulty1 = abs_tree_cnode1


# ### Finding insertable positions

# Given a rule, and the faulty symbol, the positions in the rule where the fault can be inserted are all the non-terminals that will eventually reach the symbol of the faulty symbol. That is, if we have `<digit> + <expr>` as the expansion and the faulty symbol is `<factor*>` then, since `<digit>` can never reach `<factor>`, `0` is out, and so is `1` since it is a terminal symbol. Hence, only `<expr>` remains, which when expanded, one of the expansion paths will include a `<factor>`. Hence, here `[2]` is the answer.

# In[181]:


def get_reachable_positions(rule, fkey, reachable):
    positions = []
    for i, token in enumerate(rule):
        if not is_nt(token): continue
        if fkey in reachable[token]:
            positions.append(i)
    return positions


# In[182]:


reachable1 = reachable_dict(EXPR_GRAMMAR)


# In[183]:


get_ipython().run_cell_magic('top', '', "for k in EXPR_GRAMMAR:\n    print(k)\n    for rule in EXPR_GRAMMAR[k]:\n        v = get_reachable_positions(rule, '<factor>', reachable1)\n        print('\\t', rule, v)")


# ### Insert into key definition

# The essential idea is to make the rules in the grammar such that there is one fault position in each position.
# Take one rule at a time. For each token in the rule, get the reachable tokens. If the fsym is not in reachable tokens, then the falt cannot be inserted in that position. So get all positions for the rule that we can insert fsym in, and for each position, change the symbol for later insertion.
# 
# #### Proof
# 
# The proof that this works is as follows:
# 
# Any key named $<\overline{X}>$ is guaranteed to contain at least one fault. The definition of any key by construction is, $<\overline{X}> \models \overline{R_1} \lor \overline{R_2} \ldots $ where each $\overline{R_i}$ is a rule that contains at least one $<\overline{Y}>$ such as $\overline{R_i} = \{<\overline{P}> \cdot Q \cdot R\} \lor \{P \cdot <\overline{Q}> \cdot R\} \lor \{ P \cdot Q \cdot <\overline{R}> \}$.

# In[184]:


from enum import Enum


# In[185]:


class FKey(str, Enum):
    #negate = 'NEGATE'
    #fault = 'FAULT' # not used
    atmost = 'ATMOST'
    atleast = 'ATLEAST'
    exactly = 'EXACTLY'


# In[186]:


def to_fkey_prefix(name, prefix, kind):
    #if kind == FKey.negative:
    #    return "<%s -%s>" % (name[1:-1], prefix)
    #if kind == FKey.fault: # not used
    #    return "<%s F%s>" % (name[1:-1], prefix)
    if kind == FKey.atmost:
        return "<%s *F%s>" % (name[1:-1], prefix)
    elif kind == FKey.atleast:
        return "<%s +F%s>" % (name[1:-1], prefix)
    elif kind == FKey.exactly:
        return "<%s .F%s>" % (name[1:-1], prefix)
    assert False


# In[187]:


def insert_atleast_one_fault_into_key(grammar, key, fsym, prefix, reachable):
    rules = grammar[key]
    my_rules = []
    for rule in grammar[key]:
        positions = get_reachable_positions(rule, fsym, reachable)
        if not positions: # make it len(positions) >= n if necessary
            # skip this rule because we can not embed the fault here.
            continue
        else:
            # at each position, insert the fsym
            for pos in positions:
                new_rule = [to_fkey_prefix(t, prefix, FKey.atleast)
                            if pos == p else t for p,t in enumerate(rule)]
                my_rules.append(new_rule)
    return (to_fkey_prefix(key, prefix, FKey.atleast), my_rules)


# In[188]:


get_ipython().run_cell_magic('top', '', "for key in EXPR_GRAMMAR:\n    fk, rules = insert_atleast_one_fault_into_key(EXPR_GRAMMAR, key, '<factor>', '1', reachable1)\n    print(fk)\n    for r in rules:\n        print('    ', r)\n    print()")


# In[189]:


def insert_atleast_one_fault_into_grammar(grammar, fsym, prefix_f, reachable):
    new_grammar = {}
    for key in grammar:
        fk, rules = insert_atleast_one_fault_into_key(grammar, key, fsym, prefix_f, reachable)
        if fk not in new_grammar:
            new_grammar[fk] = []
        if not rules:
            rules = grammar[key] # no applicable rules, so use the original
        new_grammar[fk].extend(rules)
    return new_grammar


# ### Get the final grammar

# The final steps are as follows:
# 1. Add the fault node, and the child nodes to the grammar.
# 2. Generate the faulty key definitions. This is done per key in the original grammar.
# 3. Finally, connect the faulty key and fault node.

# In[190]:


def atleast_one_fault_grammar(grammar, start_symbol, fault_node, f_idx):
    def L_prefix(i): return str(i)
    def F_prefix(i): return str(i)
    prefix_l = L_prefix(f_idx)
    prefix_f = F_prefix(f_idx)
    key_f = fault_node[0]
    assert key_f in grammar

    # First, get the pattern grammar
    pattern_g, pattern_s = faulty_node_to_pattern_grammar(fault_node, prefix_l)
    # the pattern grammar contains the faulty keys and their definitions.

    # Next, get the reaching grammar. This simply embeds at one guaranteed fault 
    # in each of the rules.
    reachable_keys = reachable_dict(grammar)
    # We want to insert the fault prefix_f into each insertable positions. 
    # the insertable locations are those that can reach reaching_fsym
    reach_g = insert_atleast_one_fault_into_grammar(grammar, key_f, prefix_f, reachable_keys)

    # now, the faulty key is an alternative to the original.
    # We have to take care of one thing though. The `fkey` in the pattern grammar should
    # be replaced with reaching_fsym, but the definitions kept. This is because we want to preserve
    # the rule patterns. We do not want normal expansions to go through since it may mean
    # no fault inserted. However, we want self recursion to happen.
    reaching_fsym = to_fkey_prefix(key_f, prefix_f, FKey.atleast)

    # How do we insert the fault into the grammar? Essentially, at some point we want
    # to allow reach_g[reaching_fsym] to produce the fault. If this token is
    # nonrecursive, then it simple. We replace the definition reach_g[reaching_fsym]
    # with that of pattern_g[pattern_s] and we are done.
    pattern_rule = pattern_g[pattern_s][0] # get the pattern rule
    
    # However, if the reach_g[reaching_fsym] rules contain any tokens that can
    # reach `reaching_fsym` then it becomes more complex because we do not want
    # to miss out on these patterns. On the other hand, we also need to make sure that we do
    # not introduce the fault by matching the first expansion of the fault node.
    
    # print('WARNING: atleast_one_fault_grammar is incomplete.')
    reaching_rules = []
    for rule in reach_g[reaching_fsym]:
        if normalized_rule_match(rule, pattern_rule):
            # # TODO: If this was insert _only_ one fault then,
            # # we do not want to inadvertantly introduce the fault again. So this requires special
            # # handling. In effect, we want to make sure that the rule is actually a negation
            # # of the pattern_rule, one token at a time. This can be done only once negation comes in.
            
            # However, given that we have no restriction on the number of faults inserted,
            # we can merrily add this rule. The only restriction being that, the inserted rule
            # should not allow a non-matching parse to go forward. However, this is done by
            # construction since we are using reaching rules.
            
            # # note: this may not be correct. in Factor +F1, (expr L1_1) is finite, and will
            # # get removed by gc as being more refined than (expr +F1).
            reaching_rules.append(rule)
            pass
        else:
            # we only want to keep a rule if at least one of the tokens is reaching fault_node[0].
            for token in rule_to_normalized_rule(rule):
                if is_nt(token) and key_f in reachable_keys[token]:
                    reaching_rules.append(rule)
                    break
 
    combined_grammar = {**grammar, **pattern_g, **reach_g}
    combined_grammar[reaching_fsym] = reaching_rules + [pattern_rule]
   
    return combined_grammar, to_fkey_prefix(start_symbol, F_prefix(f_idx), FKey.atleast)


# In[191]:


get_ipython().run_line_magic('top', 'node_faulty1[0]')


# The guarantee is at least one fault per input.

# In[192]:


#%%top
faulty1_grammar_, faulty1_start = atleast_one_fault_grammar(EXPR_GRAMMAR, EXPR_START, node_faulty1, 1)


# In[193]:


get_ipython().run_line_magic('top', 'Gs(faulty1_grammar_, -1)')


# In[194]:


#%%top
faulty1_grammar, faulty1_start = grammar_gc(faulty1_grammar_, faulty1_start)


# In[195]:


get_ipython().run_cell_magic('top', '', 'Gs(faulty1_grammar)\nfaulty1_start')


# In[196]:


#%%top
faulty1_fuzzer = LimitFuzzer(faulty1_grammar)
faulty1_parser = Parser(faulty1_grammar, canonical=True, start_symbol=faulty1_start)


# In[197]:


get_ipython().run_cell_magic('top', '', "for i in range(10):\n    s = faulty1_fuzzer.fuzz(key=faulty1_start)\n    print(s)\n    assert faulty1_parser.can_parse(s)\n    assert '((' in s and '))' in s")


# A few parses

# In[198]:


get_ipython().run_line_magic('top', "assert faulty1_parser.can_parse('((2))')")


# In[199]:


get_ipython().run_line_magic('top', "assert faulty1_parser.can_parse('((1 + 1))')")


# In[200]:


get_ipython().run_line_magic('top', "assert not faulty1_parser.can_parse('1 + 2')")


# In[201]:


get_ipython().run_line_magic('top', "assert faulty1_parser.can_parse('1 + ((3))')")


# ## Removing a fault

# ### Negated pattern grammar 

# Given a pattern grammar and the correspoinding grammar, we produce a negated pattern grammar for it.

# In[202]:


def negate_prefix(prefix):
    assert ' ' not in prefix
    return 'neg(%s)' % prefix


# In[203]:


def is_negative_key(key):
    nk = refinement(key)
    return nk[0:4] == 'neg('


# In[204]:


get_ipython().run_line_magic('top', "assert is_negative_key('<key neg(F1)>')")


# In[205]:


def negate_base_key(k, prefix):
    assert is_nt(k)
    assert is_base_key(k)
    return '<%s %s>' % (stem(k), negate_prefix(prefix))


# In[206]:


get_ipython().run_line_magic('top', "assert negate_base_key('<key>','F1') == '<key neg(F1)>'")


# In[207]:


def negate_key(k):
    assert is_nt(k)
    assert refinement(k)
    return '<%s %s>' % (stem(k), negate_prefix(refinement(k)))


# In[208]:


get_ipython().run_line_magic('top', "assert negate_key('<key F1>') == '<key neg(F1)>'")


# In[209]:


def unnegate_key(k):
    assert is_negative_key(k)
    ref = refinement(k)
    #neg()
    return '<%s %s>' % (stem(k), ref[4:-1])


# In[210]:


get_ipython().run_line_magic('top', "assert unnegate_key('<key neg(F1)>')  == '<key F1>'")


# In[211]:


def negate_key_at(rule, at):
    new_rule = []
    for i,key in enumerate(rule):
        if i == at:
            new_rule.append(negate_key(key))
        else:
            new_rule.append(key)
    return new_rule


# In[212]:


def negate_a_base_rule_wrt_fault_in_pattern_grammar(base_rule, fault_key, reachable_keys, log=False):
    assert reachable_keys is not None
    assert not [k for k in base_rule if is_nt(k) and is_refined_key(k)]
    # when we want to negate a base rule, we only produce a single negated rule with
    # _all_ reachable points negated. This is because if any of these points allow reach
    # of the fault, then fault can be present.
    refinements = []
    negated_rule = []
    for i, token in enumerate(base_rule):
        if not is_nt(token):
            negated_rule.append(token)
        elif normalize(fault_key) in reachable_keys[token]:
            t = negate_base_key(token, refinement(fault_key))
            refinements.append(t)
            negated_rule.append(t)
        else:
            negated_rule.append(token)
    return negated_rule, refinements


# In[213]:


get_ipython().run_cell_magic('top', '', "nrule, refs = negate_a_base_rule_wrt_fault_in_pattern_grammar(\n    ['(', '<expr>',')'], '<factor F1>', reachable_dict(EXPR_GRAMMAR))\nassert nrule == ['(', '<expr neg(F1)>', ')']\nassert refs == ['<expr neg(F1)>']")


# In[214]:


get_ipython().run_cell_magic('top', '', "nrule, refs = negate_a_base_rule_wrt_fault_in_pattern_grammar(\n    ['<term>', '+', '<expr>'], '<factor F1>', reachable_dict(EXPR_GRAMMAR))\nassert nrule == ['<term neg(F1)>', '+', '<expr neg(F1)>']")


# In[215]:


get_ipython().run_cell_magic('top', '', "nrule, refs = negate_a_base_rule_wrt_fault_in_pattern_grammar(\n    ['<integer>', '.', '<integer>'], '<factor F1>', reachable_dict(EXPR_GRAMMAR))\nassert nrule == ['<integer>', '.', '<integer>']")


# How to negate a refined rule? The easiest part is terminals. They come as they are. Next, the refined keys. They get negated. Now, for unrefined keys. In a linear grammar, there should not be any. However, what we have are not strictly linear. There are abstractions involved.
# 
# These can generate the _original_ charecteristic key. So, they need to be negated based on that.

# The difference between `negate_a_refined_rule` and `negate_a_refined_rule_in_pattern_grammar` is that we know the fault_key (the charecteristic fault). So, we can check reachability.

# In[216]:


def negate_a_refined_rule_in_pattern_grammar(refined_rule, fault_key, reachable_keys, log=False):
    assert reachable_keys is not None
    # TODO: check whether the rule is unrefined, and return early?
    
    # first, preprocess the rule
    prefix = refinement(fault_key)
    refinements = []
    skip = []
    # First, we refine our rule. Essentially, here, we negate any base keys that can reach the
    # fault. These are essentially the holes in our patterns.
    rerefined_rule = []
    for i, t in enumerate(refined_rule):
        if not is_nt(t):
            t_ = t
        elif not is_base_key(t):
            t_ = t
        # is faulty key reachable from the base key? If so, then we need to negate the
        # base key.
        elif normalize(fault_key) in reachable_keys[t]:
            t_ = negate_base_key(t, prefix)
            # The idea is to explode the expression that we want to evaluate to DNF, and
            # check whether any of the negated fault keys exist in their own negated reachability grammars.
            refinements.append(t_)
            # At this point, this is no longer a base key. So make sure to record the position.
            skip.append(i)
        else:
            t_ = t
        rerefined_rule.append(t_)

    # Now, we come to any refinements that are part of the pattern grammar. We want to produce multiple
    # rules -- as many as there are negatable tokens in the rule. Each rule will have a single position
    # negated from the refined_rule (and all base keys already negated).
    negated_rules = []
    found = False
    for i, t in enumerate(rerefined_rule):
        if not is_nt(t): continue
        if is_base_key(t): continue # note: the base key definition has changed.
        if i in skip: continue # Holes. We already negatd these.
        negated_rules.append(negate_key_at(rerefined_rule, i))
        found = True

    # if there are no refinements found, then there is nothing to negate it against.
    # which means that the match will happen if we add the rule as is. We want to prevent the
    # match. So, 
    # if not found: <- NO
    #    negated_rules.append(rerefined_rule)
    
    # e.g. '<factor L1_2>': [('(', '<expr>', ')')
    # the trouble with unrefined (i.e no refined key to negate) is that negating it is empty.
    if not found: 
        assert not [k for k in refined_rule if is_nt(k) and is_refined_key(k)]
    else:
        assert [k for k in refined_rule if is_nt(k) and is_refined_key(k)]
    return negated_rules, refinements


# In[217]:


get_ipython().run_cell_magic('top', '', "assert negate_a_refined_rule_in_pattern_grammar(\n    ['(', '<expr F1>', ')'], '<factor F1>', reachable_dict(EXPR_GRAMMAR)) == (\n    [['(', '<expr neg(F1)>', ')']], [])")


# In[218]:


get_ipython().run_cell_magic('top', '', "assert negate_a_refined_rule_in_pattern_grammar(\n    ['(', '<expr>', '+', '<expr L1>', ')'], '<factor F1>', reachable_dict(EXPR_GRAMMAR)) == (\n    [['(', '<expr neg(F1)>', '+', '<expr neg(L1)>', ')']],\n    ['<expr neg(F1)>'])")


# How do we negate a definition (a list of rules)? Essentially, any rule in the base grammar that is not part of the list should also be added (taking care to negate all the base keys).
# 
# Next, for a refined rule, we generate multiple rules from such, with one position at a time being negated to produce a new rule.

# In[219]:


def negate_definition_in_pattern_grammar(fault_key, refined_rules, base_rules, reachable_keys, log=False):
    assert reachable_keys
    refinements = []
    # the harder part. First, we find the rules in base_rules which
    # do not match the pattern in any of the refined_rules.
    # each of these rules could in principle, induce the fault again
    # except for terminals and nonreachables. So, we need to negate them.

    negated_rules_base = []
    non_matching_base_rules = rule_normalized_difference(base_rules, refined_rules)
    if log: print('> for fkey', fault_key, len(non_matching_base_rules))
    for rule in non_matching_base_rules:
        negated_rule, refs = negate_a_base_rule_wrt_fault_in_pattern_grammar(rule, fault_key, reachable_keys, log)
        negated_rules_base.append(negated_rule)
        refinements.extend(refs)
        if log: print('>  ', negated_rule)
            
    # the simple part. Given the set of fules, we take one rule at a time,
    # and genrate the negated ruleset from that.
    negated_rules_refined = []
    if log: print('> for fkey refined:', fault_key, len(refined_rules))
    for ruleR in refined_rules:
        neg_rules, refs = negate_a_refined_rule_in_pattern_grammar(ruleR, fault_key, reachable_keys, log)
        negated_rules_refined.extend(neg_rules)
        refinements.extend(refs)
        
    return negated_rules_refined + negated_rules_base, refinements


# In[220]:


get_ipython().run_line_magic('top', "faulty1_grammar['<expr +F1>']")


# In[221]:


get_ipython().run_cell_magic('top', '', "negate_definition_in_pattern_grammar('<factor +F1>', faulty1_grammar['<expr +F1>'],\n                                       EXPR_GRAMMAR['<expr>'], reachable_dict(EXPR_GRAMMAR))")


# Negating a linear grammar is fairly simple. Take a definition at a time, and negate it.

# In[222]:


def negated_pattern_grammar(pattern_grammar, pattern_start, fault_key, base_grammar, log=False):
    assert normalize(fault_key) == normalize(pattern_start)
    reachable_keys = reachable_dict(base_grammar)
    negated_grammar = {}
    refinements = []
    for l_key in pattern_grammar:
        l_rule = pattern_grammar[l_key][0]
        nl_key = negate_key(l_key)
        if log: print(l_key, '->', nl_key, ':', l_rule)
        # find all rules that do not match, and add to negated_grammar,
        # taking care to make sure reachabiity of keys.
        normal_l_key = normalize(l_key)
        base_rules = base_grammar[normal_l_key]
        refined_rules = pattern_grammar[l_key]
        
        negated_rules, refs = negate_definition_in_pattern_grammar(fault_key, refined_rules, base_rules,
                                                                reachable_keys, log)
        # TODO does negated_rules require `and` of similar rules? (see self negation)
        # The problem is that, for pattern grammar, there is only a single rule.

        negated_grammar[nl_key] = negated_rules
        refinements.extend(refs)
    # these are all negations in the original grammar. They will come later.
    return negated_grammar, negate_key(pattern_start), refinements


# In[223]:


get_ipython().run_cell_magic('top', '', "for k in lg1:\n    k1 = normalize(k)\n    print(k1)\n    for rule in EXPR_GRAMMAR[k1]:\n        print('  ', rule)")


# In[224]:


get_ipython().run_line_magic('top', 'Gs(lg1)')


# In[225]:


#%%top
nlg1, nls1, refs = negated_pattern_grammar(lg1, ls1, '<factor F1>', EXPR_GRAMMAR)


# In[226]:


get_ipython().run_cell_magic('top', '', "Gs(nlg1)\nfor r in refs:\n    print('>', r)\nnls1")


# ### Remove from key
# 
# We need to define the `remove_from_key` first. The idea is that the fault does not occur in any of the reachable nonterminals.

# In[227]:


def remove_all_instances_of_fault_from_key(grammar, key, fsym, prefix, reachable):
    ref = refinement(to_fkey_prefix(fsym, prefix, FKey.atleast))
    rules = grammar[key]
    my_rules = []
    for rule in grammar[key]:
        positions = get_reachable_positions(rule, fsym, reachable)
        if not positions: # make it len(positions) >= n if necessary
            # add this rule as is because we cannot embed the fault here.
            my_rules.append(rule)
        else:
            # at each position, insert the fsym
            new_rule = []
            for pos, token in enumerate(rule):
                if pos in positions:
                    t = negate_base_key(rule[pos], ref)
                else:
                    t = token
                new_rule.append(t)
            my_rules.append(new_rule)
    return (negate_base_key(key, ref), my_rules)


# In[228]:


get_ipython().run_cell_magic('top', '', "for key in EXPR_GRAMMAR:\n    fk, rules = remove_all_instances_of_fault_from_key(EXPR_GRAMMAR, key, '<factor>', '1', reachable1)\n    print(fk)\n    for r in rules:\n        print('    ', r)\n    print()")


# ### Remove from grammar

# In[229]:


def remove_all_instances_of_fault_from_grammar(grammar, fsym, prefix_f, reachable):
    new_grammar = {}
    for key in grammar:
        fk, rules = remove_all_instances_of_fault_from_key(grammar, key, fsym, prefix_f, reachable)
        #if not rules: continue # no applicable rules
        assert rules # there will be rules here because negation will not remove any rule.
        if fk not in new_grammar:
            new_grammar[fk] = []
        new_grammar[fk].extend(rules)
    return new_grammar


# ### Get the final grammar

# The final steps are as follows:
# 1. Add the fault node, and the child nodes to the grammar.
# 2. Generate the faulty key definitions. This is done per key in the original grammar.
# 3. Finally, connect the faulty key and fault node.

# In[230]:


def no_fault_grammar(grammar, start_symbol, fault_node, f_idx, log=False):
    def L_prefix(i): return str(i)
    def F_prefix(i): return str(i)
    prefix_l = L_prefix(f_idx)
    prefix_f = F_prefix(f_idx)
    key_f = fault_node[0]
    assert key_f in grammar
    # First, get the pattern grammar

    fsym = to_fkey_prefix(key_f, prefix_f, FKey.atleast)
    noreaching_fsym = negate_key(fsym)

    pattern_g, pattern_s = faulty_node_to_pattern_grammar(fault_node, prefix_l)
    npattern_g, npattern_s, refs = negated_pattern_grammar(pattern_g, pattern_s, fsym, grammar, log)
    reachable_keys = reachable_dict(grammar)
    # the new grammar contains the faulty keys and their definitions.
    # next, want to insert the fault prefix_f into each insertable positions. 
    # the insertable locations are those that can reach fsym
    noreach_g = remove_all_instances_of_fault_from_grammar(grammar, key_f, prefix_f, reachable_keys)
    for key in refs: assert key in noreach_g

    # now, the faulty key is an alternative to the original.
    # We have to take care of one thing though. The `fkey` in the pattern grammar should
    # be replaced with fsym, but the definitions kept. This is because we want to preserve
    # the rule patterns. We do not want normal expansions to go through since it may mean
    # no fault inserted. However, we want self recursion to happen.
    
    combined_grammar = {**grammar, **npattern_g, **noreach_g}
    new_rules = npattern_g[npattern_s] # get the negated pattern rule
    
    combined_grammar[noreaching_fsym] = new_rules
    
    return combined_grammar, negate_base_key(start_symbol, refinement(fsym))


# In[231]:


get_ipython().run_line_magic('top', 'node_faulty1[0]')


# In[232]:


#%%top
nfaulty1_grammar, nfaulty1_start = no_fault_grammar(EXPR_GRAMMAR, EXPR_START, node_faulty1, 1)


# In[233]:


get_ipython().run_cell_magic('top', '', 'Gs(nfaulty1_grammar)\n# nfaulty1_grammar, nfaulty1_start = grammar_gc(nfaulty1_grammar_, nfaulty1_start)\n# Gs(nfaulty1_grammar)\nnfaulty1_start')


# In[234]:


#%%top
nfaulty1_fuzzer = LimitFuzzer(nfaulty1_grammar)
nfaulty1_parser = Parser(nfaulty1_grammar, canonical=True, start_symbol=nfaulty1_start)


# In[235]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    s = nfaulty1_fuzzer.fuzz(key=nfaulty1_start)\n    print(s)\n    assert nfaulty1_parser.can_parse(s)\n    assert not faulty1_parser.can_parse(s)')


# A few parses

# In[236]:


get_ipython().run_line_magic('top', "assert not nfaulty1_parser.can_parse('((2))')")


# In[237]:


get_ipython().run_line_magic('top', "assert not nfaulty1_parser.can_parse('((1 + 1))')")


# In[238]:


get_ipython().run_line_magic('top', "assert nfaulty1_parser.parse('1 + 2')")


# In[239]:


get_ipython().run_line_magic('top', "assert not nfaulty1_parser.can_parse('1 + ((3))')")


# At this point, we can produce exactly one fault grammars and at most one fault grammars.

# In[ ]:





# ## At most one fault

# ### Remove except one from key

# In[240]:


def remove_all_faults_except_one_from_key(grammar, key, fsym, prefix, reachable):
    ref = refinement(to_fkey_prefix(fsym, prefix, FKey.atleast)) # negation should be atleast
    rules = grammar[key]
    my_rules = []
    for rule in grammar[key]:
        positions = get_reachable_positions(rule, fsym, reachable)
        if not positions: # make it len(positions) >= n if necessary
            # add this rule as is because we can not embed the fault here.
            my_rules.append(rule)
        else:
            # skip pos for each rule
            for pos in positions:
                new_rule = [to_fkey_prefix(t, prefix, FKey.atmost)
                            if pos == p else  # at p position, there _may be_ a fault, but not in other places
                            (negate_base_key(t, ref) if is_nt(t) else t)
                            # change to FKey.exactly to make it exactly
                            for p,t in enumerate(rule)]
                my_rules.append(new_rule)
    return (to_fkey_prefix(key, prefix, FKey.atmost), my_rules)


# In[241]:


get_ipython().run_cell_magic('top', '', "for key in EXPR_GRAMMAR:\n    fk, rules = remove_all_faults_except_one_from_key(EXPR_GRAMMAR, key, '<factor>', '1', reachable1)\n    print(fk)\n    for r in rules:\n        print('    ', r)\n    print()")


# ### Remove except one from grammar

# In[242]:


def remove_all_faults_except_one_from_grammar(grammar, fsym, prefix_f, reachable):
    new_grammar = {}
    for key in grammar:
        fk, rules = remove_all_faults_except_one_from_key(grammar, key, fsym, prefix_f, reachable)
        assert rules # there will be rules because negation is involved.
        if fk not in new_grammar:
            new_grammar[fk] = []
        new_grammar[fk].extend(rules)
    return new_grammar


# ### Get the final grammar

# In[243]:


def atmost_one_fault_grammar(grammar, start_symbol, fault_node, f_idx, log=False):
    def L_prefix(i): return str(i)
    def F_prefix(i): return str(i)
    prefix_l = L_prefix(f_idx)
    prefix_f = F_prefix(f_idx)
    key_f = fault_node[0]
    assert key_f in grammar
    # First, get the pattern grammar

    fsym = to_fkey_prefix(key_f, prefix_f, FKey.atmost)
    atleast_fsym = to_fkey_prefix(key_f, prefix_f, FKey.atleast)
    noreaching_fsym = negate_key(atleast_fsym)
    
    pattern_g, pattern_s  = faulty_node_to_pattern_grammar(fault_node, prefix_l)
    npattern_g, npattern_s, refs = negated_pattern_grammar(pattern_g, pattern_s, atleast_fsym, grammar, log)
    reachable_keys = reachable_dict(grammar)
    # the new grammar contains the faulty keys and their definitions.
    # next, want to insert the fault prefix_f into each insertable positions. 
    # the insertable locations are those that can reach fsym
    atmost_g = remove_all_faults_except_one_from_grammar(grammar, key_f, prefix_f, reachable_keys)

    noreach_g = remove_all_instances_of_fault_from_grammar(grammar, key_f, prefix_f, reachable_keys)
    negation_connect = npattern_g[npattern_s] # get the negated pattern rule
    noreach_g[noreaching_fsym] = negation_connect

    for key in refs: assert key in noreach_g, key
    
    # now, the faulty key is an alternative to the original.
    # We have to take care of one thing though. The `fkey` in the linear grammar should
    # be replaced with fsym, but the definitions kept. This is because we want to preserve
    # the rule patterns. We do not want normal expansions to go through since it may mean
    # no fault inserted. However, we want self recursion to happen.
    pattern_rule = pattern_g[pattern_s][0]
    
    combined_grammar = {**grammar, **pattern_g, **npattern_g, **atmost_g, **noreach_g}
    new_rules = pattern_g[pattern_s] # get the pattern rule
   
    combined_grammar[fsym].extend(new_rules)
    #combined_grammar[fsym] = new_rules
    return combined_grammar, to_fkey_prefix(start_symbol, F_prefix(f_idx), FKey.atmost)


# In[244]:


#%%top
nfaultya1_grammar_, nfaultya1_start = atmost_one_fault_grammar(EXPR_GRAMMAR, EXPR_START, node_faulty1, 1)


# In[245]:


get_ipython().run_cell_magic('top', '', 'Gs(nfaultya1_grammar_,-1)\nnfaultya1_start')


# In[246]:


#%%top
nfaultya1_grammar, nfaultya1_start = grammar_gc(nfaultya1_grammar_, nfaultya1_start)


# In[247]:


get_ipython().run_cell_magic('top', '', 'Gs(nfaultya1_grammar)\nnfaultya1_start')


# In[248]:


#%%top
nfaultya1_fuzzer = LimitFuzzer(nfaultya1_grammar)
nfaultya1_parser = Parser(nfaultya1_grammar, canonical=True, start_symbol=nfaultya1_start)


# In[249]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    s = nfaultya1_fuzzer.fuzz(key=nfaultya1_start)\n    print(s)\n    for t in nfaultya1_parser.parse(s):\n        assert tree_to_str(t) == s')


# A few parses

# In[250]:


get_ipython().run_line_magic('top', "assert nfaultya1_parser.can_parse('((2))')")


# In[251]:


get_ipython().run_line_magic('top', "assert nfaultya1_parser.can_parse('((1 + 1))')")


# In[252]:


get_ipython().run_line_magic('top', "assert nfaultya1_parser.can_parse('1 + 2')")


# In[253]:


get_ipython().run_line_magic('top', "assert nfaultya1_parser.can_parse('1 + ((3))')")


# In[254]:


get_ipython().run_line_magic('top', "assert not nfaultya1_parser.can_parse('((1)) + ((3))')")


# In[255]:


get_ipython().run_line_magic('top', "assert not nfaultya1_parser.can_parse('((1 + 3)) + (1) - ((2))')")


# In[256]:


get_ipython().run_line_magic('top', "assert nfaultya1_parser.can_parse('((1 + 3)) + (1) - (2 + (2))')")


# In[ ]:





# ## Exactly one fault grammar

# ### Keep exactly one at key

# In[257]:


def keep_exactly_one_fault_at_key(grammar, key, fsym, prefix, reachable):
    ref = refinement(to_fkey_prefix(fsym, prefix, FKey.atleast)) # negation should be atleast
    rules = grammar[key]
    my_rules = []
    for rule in grammar[key]:
        positions = get_reachable_positions(rule, fsym, reachable)
        if not positions: # make it len(positions) >= n if necessary
            # add this rule as is because we can not embed the fault here.
            # my_rules.append(rule)
            continue
        else:
            # skip pos for each rule
            for pos in positions:
                new_rule = [to_fkey_prefix(t, prefix, FKey.exactly)
                            if pos == p else  # at p position, there _may be_ a fault, but not in other places
                            (negate_base_key(t, ref) if is_nt(t) else t)
                            # change to FKey.exactly to make it exactly
                            for p,t in enumerate(rule)]
                my_rules.append(new_rule)
    return (to_fkey_prefix(key, prefix, FKey.exactly), my_rules)


# In[258]:


get_ipython().run_cell_magic('top', '', "for key in EXPR_GRAMMAR:\n    fk, rules = keep_exactly_one_fault_at_key(EXPR_GRAMMAR, key, '<factor>', '1', reachable1)\n    print(fk)\n    for r in rules:\n        print('    ', r)\n    print()")


# ### Keep exactly one at grammar

# In[259]:


def keep_exactly_one_fault_at_grammar(grammar, fsym, prefix_f, reachable):
    new_grammar = {}
    for key in grammar:
        fk, rules = keep_exactly_one_fault_at_key(grammar, key, fsym, prefix_f, reachable)
        if not rules: continue # no applicable rules
        if fk not in new_grammar:
            new_grammar[fk] = []
        new_grammar[fk].extend(rules)
    return new_grammar


# ### Get final grammar

# In[260]:


def exactly_one_fault_grammar(grammar, start_symbol, fault_node, f_idx, log=False):
    def L_prefix(i): return str(i)
    def F_prefix(i): return str(i)
    prefix_l = L_prefix(f_idx)
    prefix_f = F_prefix(f_idx)
    key_f = fault_node[0]
    assert key_f in grammar
    # First, get the pattern grammar

    fsym = to_fkey_prefix(key_f, prefix_f, FKey.exactly)
    atleast_fsym = to_fkey_prefix(key_f, prefix_f, FKey.atleast)
    noreaching_fsym = negate_key(atleast_fsym)
    
    pattern_g, pattern_s  = faulty_node_to_pattern_grammar(fault_node, prefix_l)
    npattern_g, npattern_s, refs  = negated_pattern_grammar(pattern_g, pattern_s, atleast_fsym, grammar, log)
    reachable_keys = reachable_dict(grammar)
    # the new grammar contains the faulty keys and their definitions.
    # next, want to insert the fault prefix_f into each insertable positions. 
    # the insertable locations are those that can reach fsym
    exactly_g = keep_exactly_one_fault_at_grammar(grammar, key_f, prefix_f, reachable_keys)
    
    noreach_g = remove_all_instances_of_fault_from_grammar(grammar, key_f, prefix_f, reachable_keys)
    negation_connect = npattern_g[npattern_s] # get the negated pattern rule
    noreach_g[noreaching_fsym] = negation_connect
    
    for key in refs: assert key in noreach_g, key
    # now, the faulty key is an alternative to the original.
    # We have to take care of one thing though. The `fkey` in the linear grammar should
    # be replaced with fsym, but the definitions kept. This is because we want to preserve
    # the rule patterns. We do not want normal expansions to go through since it may mean
    # no fault inserted. However, we want self recursion to happen.
    pattern_rule = pattern_g[pattern_s][0]

    combined_grammar = {**grammar, **pattern_g, **npattern_g, **exactly_g, **noreach_g}
    new_rules = pattern_g[pattern_s] # get the pattern rule

    #combined_grammar[fsym].extend(new_rules)
    combined_grammar[fsym] = new_rules
    return combined_grammar, to_fkey_prefix(start_symbol, F_prefix(f_idx), FKey.exactly)


# In[261]:


#%%top
efaultya1_grammar_, efaultya1_start = exactly_one_fault_grammar(EXPR_GRAMMAR, EXPR_START, node_faulty1, 1)


# In[262]:


get_ipython().run_cell_magic('top', '', 'Gs(efaultya1_grammar_, -1)')


# In[263]:


#%%top
efaultya1_grammar, efaultya1_start = grammar_gc(efaultya1_grammar_, efaultya1_start)


# In[264]:


get_ipython().run_cell_magic('top', '', 'Gs(efaultya1_grammar)\nefaultya1_start')


# In[265]:


#%%top
efaultya1_fuzzer = LimitFuzzer(efaultya1_grammar)
efaultya1_parser = Parser(efaultya1_grammar, canonical=True, start_symbol=efaultya1_start)


# In[266]:


get_ipython().run_line_magic('top', "assert efaultya1_parser.can_parse('((2))')")


# In[267]:


get_ipython().run_line_magic('top', "assert efaultya1_parser.can_parse('((1 + 1))')")


# In[268]:


get_ipython().run_line_magic('top', "assert not efaultya1_parser.can_parse('1 + 2')")


# In[269]:


get_ipython().run_line_magic('top', "assert efaultya1_parser.can_parse('1 + ((3))')")


# In[270]:


get_ipython().run_line_magic('top', "assert not efaultya1_parser.can_parse('((1)) + ((3))')")


# In[271]:


get_ipython().run_line_magic('top', "assert not efaultya1_parser.can_parse('((1 + 3)) + (1) - ((2))')")


# In[272]:


get_ipython().run_line_magic('top', "assert efaultya1_parser.can_parse('((1 + 3)) + (1) - (2 + (2))')")


# In[273]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    s = efaultya1_fuzzer.fuzz(key=efaultya1_start)\n    print(s)\n    assert efaultya1_parser.can_parse(s)')


# ## Reconstruct

# What the reconstruction does is to take any left over keys, and try to produce them based on what is available within the grammar.

# In[274]:


def remove_unused(grammar, start_symbol, log=False):
    def strip_key(grammar, key, order):
        rules = sort_rules(grammar.get(key, []))
        old_len = len(order)
        for rule in rules:
            for token in rule:
                if is_nt(token):
                    if token not in order:
                        order.append(token)
        new = order[old_len:]
        for ckey in new:
            strip_key(grammar, ckey, order)
    if start_symbol not in grammar:
        return {}, []

    order = [start_symbol]
    strip_key(grammar, start_symbol, order)
    if len(order) != len(grammar.keys()):
        stripped = [k for k in grammar if k not in order]
        faulty = [k for k in order if k not in grammar]
        if log: print('undefined:', faulty)
    new_g = {k: [list(r) for r in sort_rules(grammar[k])] for k in order if k in grammar}
    return new_g, {k:grammar[k] for k in grammar if k not in new_g}, faulty


# In[275]:


def reconstruct_rules_from_bexpr(key, tree, grammar):
    name, children = tree
    assert name == '<bexpr>'
    name_, op_ = children[0]
    bexpr = tree_to_str(tree)
    f_key = '<%s %s>' % (stem(key), bexpr)
    if f_key in grammar:
        return grammar, f_key, []
    elif name_ == '<fault>':
        # is 
        if f_key in grammar:
            return grammar, f_key, []
        else:
            nf_key = negate_key(f_key)
            assert nf_key in grammar
            fst = bexpr_parse('neg(%s)' % bexpr)
            base_grammar, base_start = normalize_grammar(grammar), normalize(key)
            g1_, s1, r1 = reconstruct_rules_from_bexpr(key, fst, grammar)
            g1, saved_keys, undef_keys  = remove_unused(g1_, s1)
            g, s, r = negate_grammar_(g1, s1, base_grammar, base_start)
            g[f_key] = g[s]
            #assert refinement(s) == bexpr
            assert s in g
            keys = find_all_nonterminals(g1)
            g = {**grammar, **g1, **g, **saved_keys}
            return g, f_key, undefined_keys(g)
    else:
        operator = op_[0][0]
        assert operator in ['and', 'or', 'neg'], operator
        assert (children[1][0],  children[-1][0]) == ('(', ')')
        new_grammar = grammar
        if operator == 'and':
            fst = children[2]
            assert children[3][0] == ','
            snd = children[4]
            if fst == snd: # or of same keys is same
                g1, s1, r1 = reconstruct_rules_from_bexpr(key, fst, grammar)
                g = {**grammar, **g1}
                g[f_key] = g[s1]
                return g, f_key, undefined_keys(g)
            g1_, s1, r1 = reconstruct_rules_from_bexpr(key, fst, grammar)
            g1, saved_keys1, undef_keys1  = remove_unused(g1_, s1)
            g2_, s2, r2 = reconstruct_rules_from_bexpr(key, snd, grammar)
            g2, saved_keys2, undef_keys2  = remove_unused(g2_, s2)
            g, s, r = and_grammars_(g1, s1, g2, s2)
            assert refinement(s) == bexpr
            assert s in g
            g = {**grammar, **g1, **g2, **g, **saved_keys1, **saved_keys2}
            return g, s, undefined_keys(g)
        elif operator == 'or':
            fst = children[2]
            assert children[3][0] == ','
            snd = children[4]
            if fst == snd: # and of same keys is same
                g1, s1, r1 = reconstruct_rules_from_bexpr(key, fst, grammar)
                g = {**grammar, **g1}
                g[f_key] = g[s1]
                return g, f_key, undefined_keys(g)
            g1_, s1, r1 = reconstruct_rules_from_bexpr(key, fst, grammar)
            g1, saved_keys1, undef_keys1  = remove_unused(g1_, s1)
            g2_, s2, r2 = reconstruct_rules_from_bexpr(key, snd, grammar)
            g2, saved_keys2, undef_keys2  = remove_unused(g2_, s2)
            g, s, r = or_grammars_(g1, s1, g2, s2)
            assert refinement(s) == bexpr
            assert s in g
            g = {**grammar, **g1, **g2, **g, **saved_keys1, **saved_keys2}
            return g, s, undefined_keys(g)
        elif operator == 'neg':
            fst = children[2]
            base_grammar, base_start = normalize_grammar(grammar), normalize(key)
            g1_, s1, r1 = reconstruct_rules_from_bexpr(key, fst, grammar)
            g1, saved_keys, undef_keys  = remove_unused(g1_, s1)
            g, s, r = negate_grammar_(g1, s1, base_grammar, base_start)
            assert refinement(s) == bexpr
            assert s in g
            keys = find_all_nonterminals(g1)
            g = {**grammar, **g1, **g, **saved_keys}
            return g, s, undefined_keys(g)
        else:
            assert False


# In[276]:


def reconstruct_key(refined_key, grammar, log=False):
    bexpr = bexpr_parse(refinement(refined_key))
    keys = [refined_key]
    defined = set()
    while keys:
        if log: print(len(keys))
        key_to_reconstruct, *keys = keys
        if log: print('reconstructing:', key_to_reconstruct)
        if key_to_reconstruct in defined: raise Exception('Key found:', key_to_reconstruct)
        defined.add(key_to_reconstruct)
        sref = simplify_bexpr(refinement(key_to_reconstruct))
        if log: print(repr(sref))
        bexpr = bexpr_parse(sref)
        nkey = '<%s %s>' % (stem(key_to_reconstruct), sref)
        if log: print('simplified_to:', nkey)
        grammar, s, refs = reconstruct_rules_from_bexpr(normalize(nkey),
                                                        bexpr, grammar)
        assert nkey in grammar
        grammar[key_to_reconstruct] = grammar[nkey]
        #for k in refs: if k not in keys: keys.append(k)
        #_, saved_keys, keys  = remove_unused(grammar, s)
        keys = undefined_keys(grammar)
        pass
    return grammar


# In[277]:


def find_reachable_keys_unchecked(grammar, key, reachable_keys=None, found_so_far=None):
    if reachable_keys is None: reachable_keys = {}
    if found_so_far is None: found_so_far = set()

    for rule in grammar.get(key, []):
        for token in rule:
            if not is_nt(token): continue
            if token in found_so_far: continue
            found_so_far.add(token)
            if token in reachable_keys:
                for k in reachable_keys[token]:
                    found_so_far.add(k)
            else:
                keys = find_reachable_keys_unchecked(grammar, token, reachable_keys, found_so_far)
                # reachable_keys[token] = keys <- found_so_far contains results from earlier
    return found_so_far


# In[278]:


def reachable_dict_unchecked(grammar):
    reachable = {}
    for key in grammar:
        keys = find_reachable_keys_unchecked(grammar, key, reachable)
        reachable[key] = keys
    return reachable


# In[279]:


def complete(grammar, start, log=False):
    keys = undefined_keys(grammar)
    reachable_keys = reachable_dict_unchecked(grammar)
    for key in keys:
        if key not in reachable_keys[start]: continue
        grammar = reconstruct_key(key, grammar, log)
    return grammar


# ## Conjunction 

# For conjunction (F1 and F2 present in the input), the idea is to simply collect all the rules that obey both patterns.
# 
# 
# The idea is simply using the distributive law. A definition is simply R1 `or` R2 `or` R3. Now, when you want to `and` two defintions, you have `and(A1 or B1 or C1, A2 or B2 or C2)` where `A1` etc are rules, and you want the `or` out again.
# 
# So, this becomes `(A1 AND A2) OR (A1 AND B2) OR (A1 AND C2) OR (A2 AND B1) OR (A2 AND C1) OR (B1 AND B2) OR (B1 AND C2) OR (B2 AND C1) OR (C1 AND C2)` which is essentially that many rules.
# 
# To construct a new rule `A1 and A2`, you simply pair each symbol in the rule up, and do `and` between them.

# In[280]:


import itertools as I


# In[281]:


def conj(k1, k2, simplify=False):
    if simplify:
        if k1 == k2:
            return k1
        elif not refinement(k1):
            return k2
        elif not refinement(k2):
            return k1
    return '<%s and(%s,%s)>' % (stem(k1), refinement(k1), refinement(k2))


# In[282]:


def and_rules(rulesA, rulesB):
    AandB_rules = []
    refinements = []
    for ruleA in rulesA:
        for ruleB in rulesB:
            if not normalized_rule_match(ruleA, ruleB): continue
            AandB_rule = []
            for t1,t2 in zip(ruleA, ruleB):
                if not is_nt(t1):
                    AandB_rule.append(t1)
                elif is_base_key(t1) and is_base_key(t2):
                    AandB_rule.append(t1)
                else:
                    k = conj(t1, t2, simplify=True)
                    refinements.append(k)
                    AandB_rule.append(k)
            AandB_rules.append(AandB_rule)
    return AandB_rules, refinements


# In[283]:


def and_grammars_(g1, s1, g2, s2):
    g1_keys = g1.keys()
    g2_keys = g2.keys()
    g = {**g1, **g2}
    refinements = []
    # now get the matching keys for each pair.
    for k1,k2 in I.product(g1_keys, g2_keys):
        # define and(k1, k2)
        if normalize(k1) != normalize(k2): continue
        # find matching rules
        and_key = conj(k1, k2) 
        g[and_key], refs = and_rules(g1[k1], g2[k2])
        refinements.extend(refs)
    #for k in refinements: assert k in g
    return g, conj(s1, s2), refinements


# In[284]:


#%%top
expr2_input = '0 + 0'
expr2_tree = list(expr_parser.parse(expr2_input))[0]


# In[285]:


#%%top
abs_path_2_a = [0,0]
abs_path_2_b = [0,2,0]


# In[286]:


get_ipython().run_line_magic('top', 'Ns(expr2_tree, [abs_path_2_a, abs_path_2_b])')


# In[287]:


#%%top
Ft2 = mark_abstract_nodes(expr2_tree, [abs_path_2_a, abs_path_2_b])


# In[288]:


get_ipython().run_line_magic('top', 'Ta(Ft2)')


# In[289]:


get_ipython().run_line_magic('top', 'Da(Ft2)')


# In[290]:


# %%top
Fp2 = find_charecterizing_node(Ft2)
faulty2_grammar_, faulty2_start  = atleast_one_fault_grammar(EXPR_GRAMMAR, EXPR_START, Fp2, '2')
faulty2_grammar, faulty2_start = grammar_gc(faulty2_grammar_, faulty2_start)


# In[291]:


get_ipython().run_line_magic('top', 'Gs(faulty2_grammar)')


# In[292]:


#%%top
faulty2_fuzzer = LimitFuzzer(faulty2_grammar)
faulty2_parser = Parser(faulty2_grammar, canonical=True, start_symbol=faulty2_start)


# In[293]:


#%%top
and1_grammar_, and1_start, refs_ = and_grammars_(faulty1_grammar, faulty1_start, faulty2_grammar, faulty2_start)


# In[294]:


get_ipython().run_line_magic('top', 'Gs(and1_grammar_, -1)')


# In[295]:


get_ipython().run_line_magic('top', 'undefined_keys(and1_grammar_)')


# In[296]:


#%%top
and1_grammar, and1_start = grammar_gc(and1_grammar_, and1_start)


# In[297]:


get_ipython().run_cell_magic('top', '', 'Gs(and1_grammar)\nand1_start')


# In[298]:


#%%top
and1_fuzzer = LimitFuzzer(and1_grammar)
and1_parser = Parser(and1_grammar, canonical=True, start_symbol=and1_start)


# In[299]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    s = and1_fuzzer.fuzz(key=and1_start)\n    print(s)\n    assert and1_parser.can_parse(s)\n    assert faulty1_parser.can_parse(s)\n    assert faulty2_parser.can_parse(s)')


# ## Disjunction
# 
# The idea here is to produce a merge of both grammars. Unlike in `and` where we combined each rule pair, we will simply add both rulesets. Note that there could be a large number of duplicate keys. This can be fixed by grammar GC.
# 
# The idea is simply using the distributive law. A definition is simply R1 `or` R2 `or` R3. Now, when you want to `or` two defintions, you have `or(A1 or B1 or C1, A2 or B2 or C2)` where `A1` etc are rules, then it simply becomes `A1 or B1 or C1 or A2 or B2 or C2`
# 

# In[300]:


def disj(k1, k2, simplify=False):
    assert is_nt(k1)
    if simplify:
        if k1 == k2:
            return k1
    return '<%s or(%s,%s)>' % (stem(k1), refinement(k1), refinement(k2))


# In[301]:


def merge_similar_rules(rule1, rule2):
    assert normalized_rule_match(rule1, rule2)
    new_rule = []
    refinements = []
    
    # note. This would not work with multi-nt-token rules.
    # so we need to restrict ourselves to grammars that only
    # allow a single token rule opposite start.
    assert len([t for t in rule1 if is_nt(t)]) == 1
    for t1,t2 in zip(rule1, rule2):
        if not is_nt(t1):
            new_rule.append(t1)
        elif is_base_key(t1) and is_base_key(t2):
            new_rule.append(t1)
        else:
            k = disj(t1, t2, simplify=True)
            new_rule.append(k)
            refinements.append((k,(t1, t2)))
    return new_rule, refinements
            


# In[302]:


def merge_similar_rules_positions(rules):
    def rule_match(rule1, rule2):
        assert normalized_rule_match(rule1, rule2)
        return [i for i, (k1, k2) in enumerate(zip(rule1, rule2)) if k1 != k2]

    def merge_rule(rule1, rule2, diffs):
        assert len(diffs) == 1
        pos = diffs[0]
        m_rule = [disj(t, rule2[i], simplify=True) if i== pos else t for i,t in enumerate(rule1)]
        return m_rule, m_rule[pos]
        
    # these rules are from same key and have similar pattern as well as similar refinement positions.
    if len(rules) == 1: return rules, []
    cur_rule, *rules = rules
    new_rules, refs = merge_similar_rules_positions(rules)
    # now check if cur_rule matches any of the new_rules. If it doesn't then add to new_rule
    # if it does, merge to the newone.
    merged_rules = []
    found = False
    for nrule in new_rules:
        diffs = rule_match(cur_rule, nrule)
        if len(diffs) <= 1:
            assert len(diffs) == 1, ("no differences(%s)" % len(diffs))
            found = True
            mrule, ref = merge_rule(cur_rule, nrule, diffs)
            refs.append(ref)
            merged_rules.append(mrule)
        else:
            merged_rules.append(nrule)
    
    if not found:
        merged_rules += [nrule]
    return merged_rules, refs


# In[303]:


def merge_disj_rules(g1):
    # what can be merged?
    # only those rules that have a single mergable refinement in the rule.
    # it is ok to have multiple refinements in a rule so long as the other
    # refinements except for the merger one is exactly the same.
    similar = {}
    for k in g1:
        for rule in g1[k]:
            ref_positions = tuple([i for i,k in enumerate(rule) if is_nt(k) and refinement(k)])
            key = (k, tuple(rule_to_normalized_rule(rule)), ref_positions)
            if key not in similar: similar[key] = set()
            similar[key].add(tuple(rule))
    # now we have similar rules that have a chance of merger.
    refinements = []
    new_grammar = {}
    for (key, rule,pos) in similar:
        srules, refs = merge_similar_rules_positions([list(r) for r in similar[(key, rule,pos)]])
        if key not in new_grammar: new_grammar[key] = []
        # same key could come in multiple times.
        new_grammar[key].extend(srules)
        refinements.extend(refs)
    return {**g1, **new_grammar}, refinements


# In[304]:


def or_grammars_(g1, s1, g2, s2):
    g = {}
    # now get the matching keys for each pair.
    for k in list(g1.keys()) + list(g2.keys()): 
         g[k] = [[t for t in r] for r in list(set([tuple(k) for k in (g1.get(k, []) + g2.get(k, []))]))]
            
    g[disj(s1, s2)] = g1[s1] + g2[s2]
    g, refinements = merge_disj_rules(g) #[s1][0], g2[s2][0])
    #new_rule, refinements = merge_similar_rules(g1[s1][0], g2[s2][0])
    #g[disj(s1, s2)] = [new_rule]
    #for (k, (t1, t2)) in refinements:
    #    g[k] = g1[t1] + g2[t2]
    return g, disj(s1, s2), refinements


# In[305]:


#%top
or1_grammar_, or1_start, refs_ =  or_grammars_(faulty1_grammar, faulty1_start, faulty2_grammar, faulty2_start)


# In[306]:


#%top
ouk = undefined_keys(or1_grammar_)


# In[307]:


get_ipython().run_line_magic('top', 'ouk')


# In[308]:


#%%top
or1_grammar_ = complete(or1_grammar_, or1_start)


# In[309]:


undefined_keys(or1_grammar_)


# In[310]:


get_ipython().run_cell_magic('top', '', 'Gs(or1_grammar_)\nor1_start')


# In[311]:


#%%top
or1_grammar, or1_start = grammar_gc(or1_grammar_, or1_start)


# In[312]:


get_ipython().run_cell_magic('top', '', 'Gs(or1_grammar)\nor1_start')


# In[313]:


#%%top
or1_fuzzer = LimitFuzzer(or1_grammar)
or1_parser = Parser(or1_grammar, canonical=True, start_symbol=or1_start)


# In[314]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    s = or1_fuzzer.fuzz(key=or1_start)\n    print(s)\n    assert or1_parser.can_parse(s)\n    if not faulty1_parser.can_parse(s):\n        assert faulty2_parser.can_parse(s)\n    elif not faulty2_parser.can_parse(s):\n        assert faulty1_parser.can_parse(s)\n    else:\n        assert and1_parser.can_parse(s)')


# ## Negation (self)

# In[315]:


EXCEPTION_HAPPENED


# Negation is approached in a fundamentally different way to inserting faults. First, we do not know what the key that when include, will cause the fault. So, there is no reachability check. On the other hand, we do know the prefix. It is the refinement of the _<start>_ symbol of the grammar to be negated.
# 
# What we need to do for any key is to negate it (consider that we do not have strict non-redundancy requirement between rules so long as they have same base pattern), which means negating the individual rules, we need to move from `neg(rule1 or rule 2 or rule3)` to `neg(rule1) and neg(rule2) and neg(rule3)` where the base patterns of three rules are same.
# 
# Now, how to negate a given rule: Take any given rule. All the base nonterminal symbols have to be negated in the sense that we have to somehow make them not reach the fault with the same prefix as `<start>`. This can be done using the reconstruct_key. Next, all refined symbols have to be negated against themselves.

# In[316]:


def_F = [
 ['<T +F1>', ' + ', '<S>', '<Q>'],
 ['<T>', ' + ', '<S +F1>', '<Q>']]


# In[317]:


ndef_F = [['<T neg(+F1)>', ' + ', '<S neg(+F1)>', '<Q>']]


# In[318]:


ndef = normalize_grammar({'<x>': def_F})['<x>']


# In[319]:


assert ndef == [('<T>', ' + ', '<S>', '<Q>')]


# In[320]:


def find_base_and_refined_positions(refined_rule):
    refined_pos = []
    base_pos = []
    refinements = []
    for i, t in enumerate(refined_rule):
        if not is_nt(t):
            t_ = t # terminal
        elif is_base_key(t):
            t_ = t
            base_pos.append(i)
        elif is_refined_key(t):
            t_ = t
            refined_pos.append(i)
        else:
            assert False
    return base_pos, refined_pos


# In[321]:


def negate_a_rule(refined_rule, log=False):
    def negate_nt(t):
        return unnegate_key(t) if is_negative_key(t) else negate_key(t)
    _base_pos, refined_pos = find_base_and_refined_positions(refined_rule)
    refinements = []
 
    # now, take refined, one at a time, and negate them, while changing all other
    # refined to base.
    negated_rules = []
    for pos in refined_pos:
        new_rule = [negate_nt(t) if pos == i else
                    (normalize(t) if i in refined_pos else t)
                    for i,t in enumerate(refined_rule)]
        negated_rules.append(new_rule)
        refinements.append(new_rule[pos])
    return negated_rules, refinements


# Negation of `def_F[0]` should result in only the refined token being negated.

# In[322]:


neg_def_F_0, rfs = negate_a_rule(def_F[0], ndef[0]);
assert neg_def_F_0 == [['<T neg(+F1)>', ' + ', '<S>', '<Q>']]


# In[323]:


nndef_F_0, rfs = negate_a_rule(ndef_F[0], ndef[0])


# In[324]:


assert nndef_F_0 == [['<T +F1>', ' + ', '<S>', '<Q>'],
                     ['<T>', ' + ', '<S +F1>', '<Q>']]


# For negating a rule, we negate all base keys at the same time, but refined keys one at a time. This is because a rule could be like  `[<1,2,3,4,5>, <6,7,8,9,0>]` and a refinement could be `[<2,4>, <6,8,0>]`. The idea of negation of a rule is that a string that gets produced by the first should not be repeated in the second. Hence, the negation of this should be `[<1,2,3,4,5>, <7,9>] | [<1,3,5>, <6,7,8,9,0>]`. That is, refinement is removed in other refined keys.

# In[325]:


def negate_ruleset(refined_rules, log=False):
    refinements = []
    # Given the set of fules, we take one rule at a time,
    # and genrate the negated ruleset from that.
    negated_rules = []
    if log: print('> refined:', len(refined_rules))
    for ruleR in refined_rules:
        neg_rules, refs = negate_a_rule(ruleR, log)
        negated_rules.append(neg_rules)
        refinements.extend(refs)
    return negated_rules, refinements


# Negation of rule set should only affect the refined keys.

# In[326]:


neg_def_F_rs, rfs = negate_ruleset(def_F)
assert neg_def_F_rs == [
    [['<T neg(+F1)>', ' + ', '<S>', '<Q>']],
    [['<T>', ' + ', '<S neg(+F1)>', '<Q>']]]


# In[327]:


nndef_F, rfs = negate_ruleset(ndef_F, ndef)


# In[328]:


assert nndef_F == [[['<T +F1>', ' + ', '<S>', '<Q>'],
                    ['<T>', ' + ', '<S +F1>', '<Q>']]]


# In[329]:


def multi_and_rules(rules):
    r1 = [rules[0]]
    refinements = []
    if len(rules) == 1:
        return r1[0], []
    for r in rules[1:]:
        r1, refs = and_rules(r1, [r])
        assert len(r1) == 1
        refinements.extend(refs)
    assert len(r1) == 1
    return r1[0], refinements


# Negate definition

# In[330]:


def negate_definition(refined_rules, base_rules, log):
    new_rulesB = []
    new_refs = []
    for base_rule in base_rules:
        # What happens when there are multiple normalized rules? Then
        # we have to consider each such rule seprate.
        refined_rules_p = rules_normalized_match_to_rule(refined_rules, base_rule)
        if not refined_rules_p:
            new_rulesB.append(base_rule)
            continue
        
        neg_rulesB, refs1 = negate_ruleset(refined_rules_p, log)
        if log: print('negate_ruleset:', len(neg_rulesB))
 
        # TODO: Now, the idea is to do a `product` of each item in
        # neg_rulesB with every other item. Each item has multiple
        # rules in them, where one refinement is negated. So combining
        # that with others of similar kind using `and` should produce
        # a negated output.
        
        # Note that `refined_rules_p` are from the exact same pattern.
        # now, we have negations of all rules. We need to compute `and` of similar
        for m_rulesB in I.product(*neg_rulesB):
            r, ref = multi_and_rules(m_rulesB)
            new_rulesB.append(r)
            new_refs.extend(ref)
        
    # now, include the unmatching base rules, and append.
    return new_rulesB, refs1 + new_refs


# Negating the original rule should result in this.

# In[331]:


neg_def_F, rfs = negate_definition(def_F, ndef, False)


# In[332]:


get_ipython().run_line_magic('top', 'neg_def_F')


# In[333]:


#assert neg_def_F == [['<T neg(+F1)>', ' + ', '<S>', '<Q>'],
#                     ['<T>', ' + ', '<S neg(+F1)>', '<Q>']]


# base:
# ```
# [(1|2) (1|2)]
# ```
# 
# refined:
# ```
# [(1)  (1|2)]   --> 1,2
# [(1|2) (1)     --> 1,1
# ```
# 
# negated:
# ```
# [(2) (1|2)]
# [(1|2) (2)     --> 1,2
# ```

# In[334]:


assert neg_def_F == [['<T neg(+F1)>', ' + ', '<S neg(+F1)>', '<Q>']]


# Now, negating the negation should result in the fault back

# In[335]:


neg_neg_def_F, rfs = negate_definition(neg_def_F, ndef, True)


# In[336]:


neg_neg_def_F


# IDEA: Do not and rules produced from exploding the same rule. Instead, pair them up.

# In[337]:


def_F


# In[338]:


assert neg_neg_def_F == def_F


# In[339]:


def negate_grammar_(refined_grammar, refined_start, base_grammar, base_start, log=False):
    fault_key = refined_start
    combined_g = copy_grammar(base_grammar)
    refinements = []
    for r_key in refined_grammar:
        if log: print('>>', r_key)
        combined_g[r_key] = refined_grammar[r_key]
        if is_base_key(r_key):
            # the same negation key can occur either from negating a refined
            # key, or from adding a negation to a base key.
            # The problem is, refined key may contain more info like pattern
            # grammar
            dk = negate_base_key(r_key, refinement(fault_key))
            if dk in combined_g:
                # already defined?
                continue
            # there is no refinement in this case. Hence,
            combined_g[dk] = []
            for rule in base_grammar[r_key]: # base and refined should be the same here.
                _, _refined_pos  = find_base_and_refined_positions(rule)
                assert not _refined_pos
                combined_g[dk].append(rule)
            continue
        elif is_negative_key(r_key) and is_base_key(unnegate_key(r_key)):
            # unk = unnegate_key(r_key)
            # TODO: check if unk is reachable from refined_grammar, and it exists in refined grammar
            #assert unk in refined_grammar, unk # if not, make up.
            continue
        dk = negate_key(r_key)
        nk = normalize(r_key)
        if log: print('gdefine[:', dk)
        # r_key should actually be faulty_key, but the effect is felt only on the `refs`.
        rules,refs = negate_definition(refined_grammar[r_key], base_grammar[nk], log)
        # special case: How do we handle unreachable refined keys that are however defined
        # with the same rules as unrefined keys? i.e refinement is zero TODO -- verify
        if rules:
            combined_g[dk] = rules
        else: # no refinement
            combined_g[dk] = refined_grammar[r_key]
        refinements.extend(refs)
        if log: print('gdefined]:', dk, combined_g[dk])
    return combined_g, negate_key(refined_start), refinements


# In[340]:


get_ipython().run_cell_magic('top', '', 'Gs(faulty1_grammar)')


# In[341]:


#%%top
# faulty1_grammar is an atleast one fault grammar.
negfaulty1_grammar_, negfaulty1_start, r_ = negate_grammar_(faulty1_grammar, faulty1_start, EXPR_GRAMMAR, EXPR_START)


# In[342]:


get_ipython().run_line_magic('top', 'Gs(negfaulty1_grammar_,-1)')


# In[343]:


#%%top
ouk = undefined_keys(negfaulty1_grammar_)


# In[344]:


get_ipython().run_line_magic('top', 'ouk')


# In[345]:


#%%top
negfaulty1_grammar_ = complete(negfaulty1_grammar_, negfaulty1_start)


# In[346]:


get_ipython().run_cell_magic('top', '', 'Gs(negfaulty1_grammar_,-1)\nnegfaulty1_start ')


# In[347]:


#%%top
negfaulty1_grammar, negfaulty1_start = grammar_gc(negfaulty1_grammar_, negfaulty1_start)


# In[348]:


get_ipython().run_cell_magic('top', '', 'Gs(negfaulty1_grammar)\nnegfaulty1_start')


# In[349]:


#%%top
negfaulty1_fuzzer = LimitFuzzer(negfaulty1_grammar)
negfaulty1_parser = Parser(negfaulty1_grammar, canonical=True, start_symbol=negfaulty1_start)


# In[350]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    s = negfaulty1_fuzzer.fuzz(key=negfaulty1_start)\n    print(s)\n    assert negfaulty1_parser.can_parse(s)\n    assert not faulty1_parser.can_parse(s)')


# In[351]:


get_ipython().run_cell_magic('top', '', "v = '((1))'\nassert not negfaulty1_parser.can_parse(v)")


# In[352]:


#%%top
negnegfaulty1_grammar_, negnegfaulty1_start,_ = negate_grammar_(negfaulty1_grammar, negfaulty1_start, EXPR_GRAMMAR, EXPR_START)


# In[353]:


undefined_keys(negnegfaulty1_grammar_)


# In[354]:


#%%top
negnegfaulty1_grammar_ = complete(negnegfaulty1_grammar_, negnegfaulty1_start)


# In[355]:


#%%top
negnegfaulty1_grammar, negnegfaulty1_start = grammar_gc(negnegfaulty1_grammar_, negnegfaulty1_start)


# In[356]:


get_ipython().run_cell_magic('top', '', 'Gs(negnegfaulty1_grammar)\nnegnegfaulty1_start')


# In[357]:


#%%top
negnegfaulty1_fuzzer = LimitFuzzer(negnegfaulty1_grammar)
negnegfaulty1_parser = Parser(negnegfaulty1_grammar, canonical=True, start_symbol=negnegfaulty1_start)


# In[358]:


assert negnegfaulty1_parser.can_parse('((1))')


# In[359]:


assert not negnegfaulty1_parser.can_parse('1')


# In[360]:


#%%top
negfaulty2_grammar_, negfaulty2_start,_ = negate_grammar_(faulty2_grammar, faulty2_start, EXPR_GRAMMAR, EXPR_START)


# In[361]:


get_ipython().run_cell_magic('top', '', 'Gs(negfaulty2_grammar_,-1)\nnegfaulty2_start ')


# In[362]:


#%%top
ouk = undefined_keys(negfaulty2_grammar_); ouk


# In[363]:


#%%top
negfaulty2_grammar_ = complete(negfaulty2_grammar_, negfaulty2_start)


# In[364]:


get_ipython().run_line_magic('top', 'Gs(negfaulty2_grammar_, -1)')


# In[365]:


get_ipython().run_line_magic('top', 'Da(Ft2)')


# In[366]:


Gs(faulty2_grammar)


# In[367]:


get_ipython().run_cell_magic('top', '', 'negfaulty2_grammar, negfaulty2_start = grammar_gc(negfaulty2_grammar_, negfaulty2_start)\nGs(negfaulty2_grammar)\nnegfaulty2_start')


# In[368]:


get_ipython().run_cell_magic('top', '', 'negfaulty2_fuzzer = LimitFuzzer(negfaulty2_grammar)\nnegfaulty2_parser = Parser(negfaulty2_grammar, canonical=True, start_symbol=negfaulty2_start)')


# In[369]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    s = negfaulty2_fuzzer.fuzz(key=negfaulty2_start)\n    print(s)\n    assert negfaulty2_parser.can_parse(s)\n    assert not faulty2_parser.can_parse(s)')


# In[370]:


get_ipython().run_cell_magic('top', '', 'negfaulty3_grammar_, negfaulty3_start,_ = negate_grammar_(and1_grammar_, and1_start, EXPR_GRAMMAR, EXPR_START)\nGs(negfaulty3_grammar_)\nnegfaulty3_start ')


# In[371]:


get_ipython().run_cell_magic('top', '', 'ouk = undefined_keys(negfaulty3_grammar_); ouk')


# In[372]:


get_ipython().run_cell_magic('top', '', 'negfaulty3_grammar_ = complete(negfaulty3_grammar_, negfaulty3_start, log=True)')


# In[373]:


get_ipython().run_cell_magic('top', '', 'negfaulty3_grammar, negfaulty3_start = grammar_gc(negfaulty3_grammar_, negfaulty3_start)')


# In[374]:


get_ipython().run_cell_magic('top', '', 'Gs(negfaulty3_grammar)\nnegfaulty3_start')


# In[375]:


get_ipython().run_cell_magic('top', '', 'negfaulty3_fuzzer = LimitFuzzer(negfaulty3_grammar)\nnegfaulty3_parser = Parser(negfaulty3_grammar, canonical=True, start_symbol=negfaulty3_start)')


# In[376]:


get_ipython().run_line_magic('top', 'is_cfg_empty(negfaulty3_grammar, negfaulty3_start)')


# In[377]:


get_ipython().run_line_magic('top', "assert negfaulty3_parser.can_parse('1')")


# In[378]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    # this is negate and(faulty1, faulty2) == or(neg(faulty1), neg(faulty2))\n    s = negfaulty3_fuzzer.fuzz(key=negfaulty3_start)\n    print(s)\n    assert negfaulty3_parser.can_parse(s)\n    assert negfaulty1_parser.can_parse(s) or negfaulty2_parser.can_parse(s)\n    assert not (faulty1_parser.can_parse(s) and faulty2_parser.can_parse(s))')


# Tricks if undefined refinements remain: The references will be `and(,neg(...))` format, which means they are actually `neg(...)`. So, find `...` in the refined, and then go for a second round with that as the start key.

# In[379]:


EXCEPTION_HAPPENED


# ## Difference
# 
# Need to identify and remove duplicate keys. Note that our `partial orders` are still primitive. We now have the machinery to do it right.

# In[380]:


def difference_grammars(grammarA, startA, grammarB, startB):
    if is_base_key(startB):
        return {'<start>': []}, '<start>' # empty
    base_g = normalize_grammar(grammarA)
    base_s = normalize(startA) 
    negB_g, negB_s, refs = negate_grammar_(grammarB, startB, base_g, base_s)
    negB_g = complete(negB_g, negB_s)
    AminusB_g, AminusB_s, refs = and_grammars_(grammarA, startA, negB_g, negB_s)
    AminusB_g = complete(AminusB_g, AminusB_s)
    return AminusB_g, AminusB_s


# In[381]:


identify_partial_orders(or1_grammar)


# Identifying partial orders is simple once you have the machinary for `and` and `neg`. To find if a given nonterminal `A` is more refined than `B`, do `A-B`. This should be empty.

# In[382]:


def is_keyA_more_refined_than_keyB(keyA, keyB, porder, grammar):
    # essential idea of comparing two keys is this:
    # One key is smaller than the other if for any given rule in the first,
    # there exist another rule that is larger than that in the second key.
    # a rule is smaller than another if all tokens in that rule is either equal
    # (matching) or smaller than
    # the corresponding token in the other.
    
    A_B_g, A_B_s = difference_grammars(grammar, keyA, grammar, keyB)
    if is_cfg_empty(A_B_g, A_B_s): #A is smaller, so A-B should be empty.
        return True
    else:
        return False


# In[383]:


def insert_into_porder(my_key, porder, grammar):
    def update_tree(my_key, tree, grammar):
        if tree is None: return True, (my_key, [])
        k, children = tree
        if is_base_key(my_key):
            if not is_base_key(k):
                return True, (my_key, [tree])
            else:
                return False, tree
 
        v = is_keyA_more_refined_than_keyB(my_key, k, porder, grammar)
        if is_base_key(k): v = True
        # if v is unknown...
        if v: # we should go into the children
            if not children:
                #print('>', 0)
                return True, (k, [(my_key, [])])
            new_children = []
            updated = False
            for c in children:
                u, c_ = update_tree(my_key, c, grammar)
                if u: updated = True
                new_children.append(c_)
            #print('>', 1)
            return updated, (k, new_children)
        else:
            v = is_keyA_more_refined_than_keyB(k, my_key, porder, grammar)
            if v:
                #this should be the parent of tree
                #print('>', 2)
                return True, (my_key, [tree])
            else:
                # add as a sibling -- but only if we have evidence.
                if v is not None:
                    #print('>', 3)
                    return True, (k, children + [(my_key, [])])
                else:
                    return False, tree
    key = normalize(my_key)
    updated, v = update_tree(my_key, porder.get(key, None), grammar)
    if updated:
        porder[key] = v
    return updated


# In[384]:


identify_partial_orders(or1_grammar)


# In[385]:


def grammar_gc(grammar, start_symbol, options=(1,2), log=False):
    g = grammar
    po = {}
    while True:
        if 1 in options:
            if log: print('remove_empty_keys..')
            g0, empty_keys = remove_empty_keys(g)
            if log:
                for k in empty_keys:
                    print('removed:', k)
        else:
            g0, empty_keys = g, []
        for k in g0:
            for rule in g0[k]:
                for t in rule: assert type(t) is str

        if 2 in options:
            if log: print('remove_unused_keys..')
            g1, unused_keys = remove_unused_keys(g0, start_symbol)
        else:
            g1, unused_keys = g0, []
        for k in g1:
            for rule in g1[k]:
                for t in rule: assert type(t) is str

        if 3 in options:
            if log: print('remove_redundant_rules..')
            g2, redundant_rules = remove_redundant_rules(g1, po)
        else:
            g2, redundant_rules = g1, 0
        g = g2

        if log:
            print('GC: ', unused_keys, empty_keys)
        if not (len(unused_keys) + len(empty_keys) + redundant_rules):
            break
    return g, start_symbol


# In[386]:


or1_grammar, or1_start = grammar_gc(or1_grammar_, or1_start, options=(1,2,3))
Gs(or1_grammar)
or1_start


# In[387]:


and1_grammar, and1_start = grammar_gc(and1_grammar, and1_start, options=(1, 2, 3))
Gs(and1_grammar)
and1_start


# # Experiments

# ## Fault A (exactly 1)

# In[388]:


#%%top
exprA_input = '((1))'
exprA_tree = list(expr_parser.parse(exprA_input))[0]


# In[389]:


get_ipython().run_line_magic('top', 'tree_to_str(exprA_tree)')


# In[390]:


get_ipython().run_line_magic('top', 'display_tree(exprA_tree)')


# In[391]:


#%%top
abs_path_A = [0,0,0,1,0,0,1]


# In[392]:


get_ipython().run_line_magic('top', 'Ns(exprA_tree, [abs_path_A])')


# In[393]:


#%%top
FtA = mark_abstract_nodes(exprA_tree, [abs_path_A])


# In[394]:


get_ipython().run_line_magic('top', 'Ta(FtA)')


# In[395]:


get_ipython().run_line_magic('top', 'Da(FtA)')


# In[396]:


#%%top
FpA = find_charecterizing_node(FtA)
faultA_grammar_, faultA_start = exactly_one_fault_grammar(EXPR_GRAMMAR, EXPR_START, FpA, 'A')


# In[397]:


get_ipython().run_line_magic('top', 'Gs(faultA_grammar_, -1)')


# In[398]:


#%%top
faultA_grammar, faultA_start = grammar_gc(faultA_grammar_, faultA_start)


# In[399]:


get_ipython().run_cell_magic('top', '', 'Gs(faultA_grammar)\nfaultA_start')


# In[400]:


#%%top
faultA_fuzzer = LimitFuzzer(faultA_grammar)
faultA_parser = Parser(faultA_grammar, canonical=True, start_symbol=faultA_start)


# In[401]:


get_ipython().run_cell_magic('top', '', 'for i in range(10):\n    s = faultA_fuzzer.fuzz(key=faultA_start)\n    print(s)\n    assert faultA_parser.can_parse(s)')


# In[402]:


get_ipython().run_line_magic('top', "assert not faultA_parser.can_parse('1')")


# ### -A (no fault)

# In[403]:


#%%top
FpA = find_charecterizing_node(FtA)
no_faultA_grammar_, no_faultA_start = no_fault_grammar(EXPR_GRAMMAR, EXPR_START, FpA, 'A')


# In[404]:


get_ipython().run_cell_magic('top', '', 'Gs(no_faultA_grammar_, -1)')


# In[405]:


#%%top
no_faultA_grammar, no_faultA_start = grammar_gc(no_faultA_grammar_, no_faultA_start)


# In[406]:


get_ipython().run_cell_magic('top', '', 'Gs(no_faultA_grammar)\nno_faultA_start')


# In[407]:


#%%top
no_faultA_fuzzer = LimitFuzzer(no_faultA_grammar)
no_faultA_parser = Parser(no_faultA_grammar, canonical=True, start_symbol=no_faultA_start)


# In[408]:


get_ipython().run_line_magic('top', "assert no_faultA_parser.can_parse('1')")


# In[409]:


get_ipython().run_line_magic('top', "assert not no_faultA_parser.can_parse('((1))')")


# ### neg(A)

# In[410]:


#%%top
neg_A_grammar_, neg_A_start, refs = negate_grammar_(faultA_grammar_, faultA_start, EXPR_GRAMMAR, EXPR_START, log=True)


# In[411]:


get_ipython().run_cell_magic('top', '', 'Gs(neg_A_grammar_, -1)')


# In[412]:


undefined_keys(neg_A_grammar_)


# In[413]:


alA_grammar_, alA_start = atleast_one_fault_grammar(EXPR_GRAMMAR, EXPR_START, FpA, 'A')


# In[414]:


neg_A_grammar_ = complete({**neg_A_grammar_, **alA_grammar_}, neg_A_start)


# In[415]:


#%%top
neg_A_grammar, neg_A_start = grammar_gc(neg_A_grammar_, neg_A_start)


# In[416]:


Gs(neg_A_grammar)
neg_A_start


# In[417]:


Gs_s(neg_A_grammar)


# In[ ]:





# In[418]:


neg_A_fuzzer = LimitFuzzer(neg_A_grammar)
neg_A_parser = Parser(neg_A_grammar, canonical=True, start_symbol=neg_A_start)


# `neg(A)` is the exact opposite in refinement of `A` which is exactly one fault. So it should produce either no fault or more than one fault.

# In[419]:


assert neg_A_parser.can_parse('1')


# In[420]:


assert not neg_A_parser.can_parse('((1))')


# In[421]:


assert neg_A_parser.can_parse('((1)) + ((2))')


# In[422]:


for i in range(1):
    s = neg_A_fuzzer.fuzz(key=neg_A_start, max_depth=1)
    print(s)
    assert neg_A_parser.can_parse(s)
    assert not faultA_parser.can_parse(s)


# ## Fault B

# In[423]:


exprB_input = '0 + 0'
exprB_tree = list(expr_parser.parse(exprB_input))[0]
tree_to_str(exprB_tree)


# In[424]:


abs_path_B_a = [0,0]
abs_path_B_b = [0,2,0]
Ns(exprB_tree, [abs_path_B_a, abs_path_B_b])


# In[425]:


FtB = mark_abstract_nodes(exprB_tree, [abs_path_B_a, abs_path_B_b])
Ta(FtB)


# In[426]:


Da(FtB)


# In[427]:


FpB = find_charecterizing_node(FtB)
faultB_grammar_, faultB_start  = exactly_one_fault_grammar(EXPR_GRAMMAR, EXPR_START, FpB, 'B')
Gs(faultB_grammar_)
faultB_grammar, faultB_start = grammar_gc(faultB_grammar_, faultB_start)
Gs(faultB_grammar)
faultB_start


# In[428]:


faultB_fuzzer = LimitFuzzer(faultB_grammar)
faultB_parser = Parser(faultB_grammar, canonical=True, start_symbol=faultB_start)


# In[429]:


for i in range(1):
    s = faultB_fuzzer.fuzz(key=faultB_start)
    print(s)
    assert faultB_parser.can_parse(s)


# In[430]:


assert not faultB_parser.can_parse('1 - 2')


# In[ ]:





# ### -B

# In[431]:


FpB = find_charecterizing_node(FtB)
no_faultB_grammar_, no_faultB_start = no_fault_grammar(EXPR_GRAMMAR, EXPR_START, FpB, 'B')
Gs(no_faultB_grammar_, -1)
no_faultB_grammar, no_faultB_start = grammar_gc(no_faultB_grammar_, no_faultB_start)
Gs(no_faultB_grammar)
no_faultB_start


# In[432]:


no_faultB_fuzzer = LimitFuzzer(no_faultB_grammar)
no_faultB_parser = Parser(no_faultB_grammar, canonical=True, start_symbol=no_faultB_start)


# In[433]:


assert no_faultB_parser.can_parse('1 - 2')


# In[434]:


assert not no_faultB_parser.can_parse('1 + 2')


# In[ ]:





# ## Fault C

# In[435]:


exprC_input = '1 / 0'
exprC_tree = list(expr_parser.parse(exprC_input))[0]
tree_to_str(exprC_tree)


# In[436]:


abs_path_C = [0,0, 0]
Ns(exprC_tree, [abs_path_C])


# In[437]:


FtC = mark_abstract_nodes(exprC_tree, [abs_path_C])
Da(FtC)


# In[438]:


FpC = find_charecterizing_node(FtC)
faultC_grammar_, faultC_start = exactly_one_fault_grammar(EXPR_GRAMMAR, EXPR_START, FpC, 'C')
Gs(faultC_grammar_, -1)
faultC_grammar, faultC_start = grammar_gc(faultC_grammar_, faultC_start)
Gs(faultC_grammar)
faultC_start


# In[439]:


faultC_fuzzer = LimitFuzzer(faultC_grammar)
faultC_parser = Parser(faultC_grammar, canonical=True, start_symbol=faultC_start)


# In[440]:


for i in range(10):
    s = faultC_fuzzer.fuzz(key=faultC_start)
    print(s)
    assert faultC_parser.can_parse(s)


# In[441]:


assert not faultC_parser.can_parse('1 - 2')


# In[ ]:





# ### -C

# In[442]:


FpC = find_charecterizing_node(FtC)
no_faultC_grammar_, no_faultC_start = no_fault_grammar(EXPR_GRAMMAR, EXPR_START, FpC, 'C')
Gs(no_faultC_grammar_, -1)
no_faultC_grammar, no_faultC_start = grammar_gc(no_faultC_grammar_, no_faultC_start)
Gs(no_faultC_grammar)
no_faultC_start


# In[443]:


no_faultC_fuzzer = LimitFuzzer(no_faultC_grammar)
no_faultC_parser = Parser(no_faultC_grammar, canonical=True, start_symbol=no_faultC_start)


# In[444]:


assert no_faultC_parser.can_parse('1 - 2')


# In[445]:


assert not no_faultC_parser.can_parse('1 / 0')


# ## Conjunction

# ### A & B

# In[446]:


AandB_grammar_, AandB_start, refs = and_grammars_(faultA_grammar, faultA_start, faultB_grammar, faultB_start)
Gs(AandB_grammar_, -1)
AandB_grammar, AandB_start = grammar_gc(AandB_grammar_, AandB_start)
Gs(AandB_grammar)
AandB_start


# In[447]:


AandB_fuzzer = LimitFuzzer(AandB_grammar)
AandB_parser = Parser(AandB_grammar, canonical=True, start_symbol=AandB_start)


# In[448]:


for i in range(10):
    s = AandB_fuzzer.fuzz(key=AandB_start)
    print(s)
    print('A&B')
    assert AandB_parser.can_parse(s)
    print('A')
    assert faultA_parser.can_parse(s)
    print('B')
    assert faultB_parser.can_parse(s)


# ### A & C

# In[449]:


AandC_grammar_, AandC_start, refs = and_grammars_(faultA_grammar, faultA_start, faultC_grammar, faultC_start)
Gs(AandC_grammar_, -1)
AandC_grammar, AandC_start = grammar_gc(AandC_grammar_, AandC_start)
Gs(AandC_grammar)
AandC_start


# In[450]:


AandC_fuzzer = LimitFuzzer(AandC_grammar)
AandC_parser = Parser(AandC_grammar, canonical=True, start_symbol=AandC_start)


# In[451]:


for i in range(10):
    s = AandC_fuzzer.fuzz(key=AandC_start)
    print(s)
    print('A&C')
    assert AandC_parser.can_parse(s)
    print('A')
    assert faultA_parser.can_parse(s)
    print('C')
    assert faultC_parser.can_parse(s)


# ### B & C

# In[452]:


BandC_grammar_, BandC_start, refs = and_grammars_(faultB_grammar, faultB_start, faultC_grammar, faultC_start)
Gs(BandC_grammar_, -1)
BandC_grammar, BandC_start = grammar_gc(BandC_grammar_, BandC_start)
Gs(BandC_grammar)
BandC_start


# In[453]:


BandC_fuzzer = LimitFuzzer(BandC_grammar)
BandC_parser = Parser(BandC_grammar, canonical=True, start_symbol=BandC_start)


# In[454]:


for i in range(10):
    s = BandC_fuzzer.fuzz(key=BandC_start)
    print(s)
    print('B&C')
    assert BandC_parser.can_parse(s)
    print('B')
    assert faultB_parser.can_parse(s)
    print('C')
    assert faultC_parser.can_parse(s)


# ## Disjunction

# ### A | B

# In[455]:


AorB_grammar_, AorB_start, refs = or_grammars_(faultA_grammar, faultA_start, faultB_grammar, faultB_start)
Gs(AorB_grammar_, -1)


# In[456]:


ouk = undefined_keys(AorB_grammar_); ouk


# In[457]:


#%%top
AorB_grammar_ = complete(AorB_grammar_, AorB_start, log=True)


# In[458]:


AorB_grammar, AorB_start = grammar_gc(AorB_grammar_, AorB_start)
Gs(AorB_grammar)
AorB_start


# In[459]:


AorB_fuzzer = LimitFuzzer(AorB_grammar)
AorB_parser = Parser(AorB_grammar, canonical=True, start_symbol=AorB_start)


# In[460]:


for i in range(10):
    s = AorB_fuzzer.fuzz(key=AorB_start)
    print(s)
    print('A|B')
    assert AorB_parser.can_parse(s)
    assert faultA_parser.can_parse(s) or faultB_parser.can_parse(s)


# ### A | C

# In[461]:


AorC_grammar_, AorC_start, refs = or_grammars_(faultA_grammar, faultA_start, faultC_grammar, faultC_start)
Gs(AorC_grammar_, -1)


# In[462]:


ouk = undefined_keys(AorC_grammar_); ouk


# In[463]:


#%%top
AorC_grammar_ = complete(AorC_grammar_, AorC_start, log=True)


# In[464]:


AorC_grammar, AorC_start = grammar_gc(AorC_grammar_, AorC_start)
Gs(AorC_grammar)
AorC_start


# In[465]:


AorC_fuzzer = LimitFuzzer(AorC_grammar)
AorC_parser = Parser(AorC_grammar, canonical=True, start_symbol=AorC_start)


# In[466]:


for i in range(10):
    s = AorC_fuzzer.fuzz(key=AorC_start)
    print(s)
    print('A|C')
    assert AorC_parser.can_parse(s)
    assert faultA_parser.can_parse(s) or faultC_parser.can_parse(s)


# ### B | C

# In[467]:


BorC_grammar_, BorC_start, refs = or_grammars_(faultB_grammar, faultB_start, faultC_grammar, faultC_start)
Gs(BorC_grammar_, -1)


# In[468]:


ouk = undefined_keys(BorC_grammar_); ouk


# In[469]:


#%%top
BorC_grammar_ = complete(BorC_grammar_, BorC_start, log=True)


# In[470]:


BorC_grammar, BorC_start = grammar_gc(BorC_grammar_, BorC_start)
Gs(BorC_grammar)
BorC_start


# In[471]:


BorC_fuzzer = LimitFuzzer(BorC_grammar)
BorC_parser = Parser(BorC_grammar, canonical=True, start_symbol=BorC_start)


# In[472]:


for i in range(10):
    s = BorC_fuzzer.fuzz(key=BorC_start)
    print(s)
    print('B|C')
    assert BorC_parser.can_parse(s)
    assert faultB_parser.can_parse(s) or faultC_parser.can_parse(s)


# ## Negation

# ### A - B

# In[473]:


AminusB_grammar_, AminusB_start, refs = and_grammars_(faultA_grammar, faultA_start, no_faultB_grammar, no_faultB_start)
Gs(AminusB_grammar_)
AminusB_start


# In[474]:


undefined_keys(AminusB_grammar_)


# In[475]:


AminusB_grammar, AminusB_start = grammar_gc(AminusB_grammar_, AminusB_start)
Gs(AminusB_grammar)
AminusB_start


# In[476]:


AminusB_fuzzer = LimitFuzzer(AminusB_grammar)
AminusB_parser = Parser(AminusB_grammar, canonical=True, start_symbol=AminusB_start)


# In[477]:


for i in range(10):
    s = AminusB_fuzzer.fuzz(key=AminusB_start)
    print(s)
    print('A-B')
    assert AminusB_parser.can_parse(s)
    print('A')
    assert faultA_parser.can_parse(s)
    print('B')
    assert not faultB_parser.can_parse(s)


# ### A - C

# In[478]:


AminusC_grammar_, AminusC_start, refs = and_grammars_(faultA_grammar, faultA_start, no_faultC_grammar, no_faultC_start)
Gs(AminusC_grammar_, -1)


# In[479]:


AminusC_grammar, AminusC_start = grammar_gc(AminusC_grammar_, AminusC_start)
Gs(AminusC_grammar)
AminusC_start


# In[480]:


AminusC_fuzzer = LimitFuzzer(AminusC_grammar)
AminusC_parser = Parser(AminusC_grammar, canonical=True, start_symbol=AminusC_start)


# In[481]:


for i in range(10):
    s = AminusC_fuzzer.fuzz(key=AminusC_start)
    print(s)
    print('A-C')
    assert AminusC_parser.can_parse(s)
    print('A')
    assert faultA_parser.can_parse(s)
    print('C')
    assert not faultC_parser.can_parse(s)


# ### B - C

# In[482]:


BminusC_grammar_, BminusC_start, refs = and_grammars_(faultB_grammar, faultB_start, no_faultC_grammar, no_faultC_start)
Gs(BminusC_grammar_, -1)


# In[483]:


BminusC_grammar, BminusC_start = grammar_gc(BminusC_grammar_, BminusC_start)
Gs(BminusC_grammar)
BminusC_start


# In[484]:


BminusC_fuzzer = LimitFuzzer(BminusC_grammar)
BminusC_parser = Parser(BminusC_grammar, canonical=True, start_symbol=BminusC_start)


# In[485]:


for i in range(10):
    s = BminusC_fuzzer.fuzz(key=BminusC_start)
    print(s)
    print('B-C')
    assert BminusC_parser.can_parse(s)
    print('B')
    assert faultB_parser.can_parse(s)
    print('C')
    assert not faultC_parser.can_parse(s)


# ## More

# ### A & B & C

# In[486]:


AandBandC_grammar_, AandBandC_start, refs = and_grammars_(AandB_grammar, AandB_start, faultC_grammar, faultC_start)
Gs(AandBandC_grammar_, -1)
AandBandC_grammar, AandBandC_start = grammar_gc(AandBandC_grammar_, AandBandC_start)
Gs(AandBandC_grammar)
AandBandC_start


# In[487]:


AandBandC_fuzzer = LimitFuzzer(AandBandC_grammar)
AandBandC_parser = Parser(AandBandC_grammar, canonical=True, start_symbol=AandBandC_start)


# In[488]:


for i in range(10):
    s = AandBandC_fuzzer.fuzz(key=AandBandC_start)
    print(s)
    print('A&B&C')
    assert AandBandC_parser.can_parse(s)
    print('A&B')
    assert AandB_parser.can_parse(s)
    print('A&C')
    assert AandC_parser.can_parse(s)
    print('B&C')
    assert BandC_parser.can_parse(s)
    print('A')
    assert faultA_parser.can_parse(s)
    print('B')
    assert faultB_parser.can_parse(s)
    print('C')
    assert faultC_parser.can_parse(s)


# ### A | B | C

# In[489]:


AorBorC_grammar_, AorBorC_start, refs = or_grammars_(AorB_grammar_, AorB_start, faultC_grammar_, faultC_start)
Gs(AorBorC_grammar_, -1)


# In[490]:


ouk = undefined_keys(AorBorC_grammar_); ouk


# In[491]:


AorBorC_grammar_ = complete(AorBorC_grammar_, AorBorC_start, log=True)


# In[492]:


AorBorC_grammar, AorBorC_start = grammar_gc(AorBorC_grammar_, AorBorC_start)
Gs(AorBorC_grammar)
AorBorC_start


# In[493]:


AorBorC_fuzzer = LimitFuzzer(AorBorC_grammar)
AorBorC_parser = Parser(AorBorC_grammar, canonical=True, start_symbol=AorBorC_start)


# In[494]:


for i in range(10):
    s = AorBorC_fuzzer.fuzz(key=AorBorC_start)
    print(s)
    print('A|B|C')
    assert AorBorC_parser.can_parse(s)
    print('A|B')
    assert AorB_parser.can_parse(s) or faultC_parser.can_parse(s)
    print('A|C')
    assert AorC_parser.can_parse(s) or faultB_parser.can_parse(s)
    print('B|C')
    assert BorC_parser.can_parse(s) or faultA_parser.can_parse(s)
    print('*')
    assert faultA_parser.can_parse(s) or  faultB_parser.can_parse(s) or  faultC_parser.can_parse(s)


# ### A & B | C

# In[495]:


AandBorC_grammar_, AandBorC_start, refs = or_grammars_(AandB_grammar, AandB_start, faultC_grammar, faultC_start)
AandBorC_grammar_ = complete(AandBorC_grammar_, AandBorC_start, True)
Gs(AandBorC_grammar_, -1)


# In[496]:


AandBorC_grammar, AandBorC_start = grammar_gc(AandBorC_grammar_, AandBorC_start)
Gs(AandBorC_grammar)
AandBorC_start


# In[497]:


AandBorC_fuzzer = LimitFuzzer(AandBorC_grammar)
AandBorC_parser = Parser(AandBorC_grammar, canonical=True, start_symbol=AandBorC_start)


# In[498]:


for i in range(10):
    s = AandBorC_fuzzer.fuzz(key=AandBorC_start)
    print(s)
    print('A&B|C')
    assert AandBorC_parser.can_parse(s)
    print('A&B')
    assert AandB_parser.can_parse(s) or faultC_parser.can_parse(s)


# In[499]:


EXCEPTION_HAPPENED

