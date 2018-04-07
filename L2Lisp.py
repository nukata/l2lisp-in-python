#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Little Lazy Lisp 7.2 in Python                           H20.6/10 (鈴)

Python 2.3-3.0 による小さな Lisp インタープリタ

SYNOPSIS: L2Lisp.py [file ...] [-]

無引数で起動すると対話セッションに入る。引数としてファイル名を与えると
それらを Lisp スクリプトとして順に実行する。引数としてハイフンを与える
とスクリプトの実行後，対話セッションに入る。この文は Lisp 関数 (help) 
で，著作権表示は (copyright) でそれぞれ表示できる。

次のようにライブラリとしても利用できる:
  import L2Lisp as LL
  i = LL.Interp()            # Lisp インタープリタを構築
  sy_str = LL.Symbol("str")  # Lisp シンボル str を構築
  i.symbol[sy_str] = str     # Lisp 組込み関数 str を追加
  e = LL.llist(sy_str, 123)  # Lisp リスト (str 123) を構築
  print(e)                   # (str 123) を表示
  r = i.eval(e)              # (str 123) を評価 => 文字列 "123"
  r = i.run("(+ 1 2 3)")     # (+ 1 2 3) を評価 => 6

各 Lisp 値は次のように Python のオブジェクトで表現される:
  数, 文字列 => 数, 文字列
  t          => True
  nil        => NIL (実体は空タプル)
  シンボル   => Symbol 値 (大域変数値は Interp#symbol に格納される)
  cons セル  => Cell 値

Lisp リストは Cell#__iter__ により Python の列として扱われる。
Python との相互作用のために次の４関数がある:
  (python-exec s)
    文字列 s を Python の文または文の並びとして実行し，nil を返す。
  (python-eval s)
    文字列 s を Python の式として評価し，結果の値を返す。
  (python-apply fn a k)
    Python の式 fn(*a, **dict(k)) を実行し，結果の値を返す。
  (python-self)
    Python のオブジェクトとしてのインタープリタ自身を返す。

python-exec, python-eval は Iterp#python_env を環境とする。
この環境は Interp オブジェクトの構築時に空の辞書として作られる。

特徴:
* 基本的には Emacs Lisp のサブセットだが，静的スコープをとる。
* 常に末尾呼出しの最適化を行う。
* *version* は版数とプラットフォームの２要素のリストを値とする。
* 関数は数や文字列と同じく自己評価的な一級の値であり固有の名を持たない。
  スペシャルフォームは一級の値ではなく，特定のシンボルで表される。
* 関数 apply (大域変数 apply の初期値である関数。以下同様) は２引数に限る。
* 除算と減算 / と - は１個以上の引数をとる。
* 除算と剰余 / と % は負数に対し Python の演算方法に従う。
* (eval e) は e を大域的な環境で評価する。
* (eql x y) の結果は Python の x == y に従う。
* (delay x) は Scheme と同じく x の約束を作る。~x と略記できる。
  組込み関数と条件式は約束に対し implicit forcing を行う。
* (read) は EOF に対して *eof* の大域変数値を返す。
* 評価時例外 EvalError は (catch *error* …) で捕捉できる。
* 組込み関数で発生した例外は EvalError に変換されて送出される。ただし
  EvalError 自身, KeyboardInterrupt, SystemExit はそのまま送出される。
* (lambda …) を評価すると仮引数が "コンパイル" された関数が返される。
  このとき，入れ子で含まれている (lambda …) も再帰的にコンパイルされる。
* (macro …) は大域的な環境でだけ評価でき，"マクロ式" という関数が返される。
  このマクロ式を適用した時，引数は評価されず，適用結果が再び評価される。
* マクロ式の中の自由なシンボルは捕捉されないが，マクロ引数は捕捉され得る。
* (macro …) 内の $ で始まるシンボルは dummy symbol と解釈される。
  dummy symbol は自己評価的であり，そのマクロ式の中でだけ eq が成り立つ。
* (lambda …) の評価時，最大 MAX_MACRO_EXPS 重だけ再帰的にマクロ展開する
  (非大域的に束縛されたマクロを除く)。残りは適用時に処理される。
* 印字する時，高々 MAX_EXPANSIONS 重だけ再帰的に印字済みリストを印字する。
* (dump) は Interp#symbol のキーと環境のリストを返す。
* read 時の字句解析で文字列トークンは Python の文字列として評価される。
  文字列トークン内の \n などのエスケープ列はこのとき解釈される。
* 準引用のバッククォート，カンマ，カンマアットは read 時に解決される。
  例: '`((,a b) ,c ,@d) => (cons (list a 'b) (cons c d))

スペシャルフォーム:
  quote progn cond setq lambda macro catch unwind-protect delay
組込み関数:
  car cdr cons atom stringp numberp eq eql list
  prin1 princ terpri read + * / - % <
  load eval apply force rplaca rplacd throw mapcar mapc length
  python-exec python-eval python-apply python-self
  dump help prelude copyright (および concat 実装用の _add _concat)
組込み変数:
  *error* *version* *eof*

Lisp 自身による標準の定義 (defun, if を含む) は (prelude) で表示できる。

URL: http://www.okisoft.co.jp/esc/llsp/
"""

COPYRIGHT = """
Copyright (c) 2007, 2008 Oki Software Co., Ltd.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without 
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND  
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

__version__ = 7.2

import operator, re, sys, threading

try:
    _NUMS = (int, float, complex, long) # for Python 2.3-2.6
    from cStringIO import StringIO
except NameError:
    _NUMS = (int, float, complex) # for Python 3.0
    from io import StringIO
    raw_input = input

try:
    set
except NameError:
    import sets                 # for Python 2.3
    set = sets.Set

try:
    import readline             # raw_input での行編集を可能にする
except ImportError:
    pass

NIL = ()                        # Lisp の nil 値

MAX_EXPANSIONS = 5              # 再帰的に印字する深さ
MAX_MACRO_EXPS = 20             # 静的にマクロ展開する深さ
MAX_EXC_TRACES = 10             # 例外発生時の評価トレースの記録段数

SYMBOL_MAP = {}           # 印字名からシンボルへ一意に変換するための表
SYMBOL_LOCK = threading.Lock()  # シンボル構築時の排他ロック


class Symbol (object):
    "シンボル: これ自体には変数としての値はない"
    __slots__ = '__name'

    def __new__(cls, name):
        "シンボルの印字名を引数として構築する。印字名が同じなら同じ値を返す"
        SYMBOL_LOCK.acquire()
        try:
            self = SYMBOL_MAP.get(name)
            if self is None:
                self = object.__new__(cls)
                self.__name = name
                SYMBOL_MAP[name] = self
            return self
        finally:
            SYMBOL_LOCK.release()

    def __repr__(self):
        return 'Symbol(%r)' % self.__name

    def __str__(self):
        return self.__name


S_APPEND = Symbol('append')
S_CATCH = Symbol('catch')
S_COND = Symbol('cond')
S_CONS = Symbol('cons')
S_DELAY = Symbol('delay')
S_EOF = Symbol('#<eof>')
S_ERROR = Symbol('*error*')
S_LAMBDA = Symbol('lambda')
S_LIST = Symbol('list')
S_MACRO = Symbol('macro')
S_PROGN = Symbol('progn')
S_QUOTE = Symbol('quote')
S_REST = Symbol('&rest')
S_SETQ = Symbol('setq')
S_UNWIND_PROTECT = Symbol('unwind-protect')


class SyntaxError (Exception):
    "Lisp 式の構文エラー"
    pass


class EvalError (Exception):
    "Lisp 式評価時の例外。Lisp 式の呼出し trace を持つ"
    __slots__ = 'trace'

    def __init__(self, message, expression=None):
        self.args = (message, expression)
        self.trace = []

    def __str__(self):
        (msg, exp) = self.args
        if exp is not None:
            msg += ': ' + lstr(exp)
        s = ['*** ' + msg]
        for (index, t) in enumerate(self.trace):
            s.append('%3d: %s' % (index, t))
        return '\n'.join(s)


class Thrown (EvalError):
    "Lisp の (throw tag value) の実装のための例外"
    __slots__ = 'tag', 'value'

    def __init__(self, tag, value):
        msg = 'no catcher for (%s %s)' % (lstr(tag), lstr(value))
        EvalError.__init__(self, msg)
        self.tag = tag
        self.value = value


def VariableExpected(exp):
    return EvalError('variable expected', exp)

def ProperListExpected(exp):
    return EvalError('proper list expected', exp)


class Cell (object):
    "cons セル"
    __slots__ = 'car', 'cdr'

    def __init__(self, car, cdr):
        "Lisp の (cons car cdr) に相当"
        self.car = car
        self.cdr = cdr

    def __repr__(self):
        return 'Cell(%r, %r)' % (self.car, self.cdr)

    def __str__(self):
        return lstr(self)

    def __iter__(self):
        "Lisp のリストとして各要素を与える"
        j = self
        while isinstance(j, Cell):
            yield j.car
            j = j.cdr
            if isinstance(j, Promise):
                j = j.deliver()
        if j is not NIL:
            raise ProperListExpected(j)

    def __len__(self):
        "Lisp のリストとしての長さ"
        c = 0
        for e in self:
            c += 1
        return c

    def _lrepr(self, print_quote, reclevel, printed):
        "関数 lstr のための補助メソッド"
        if self in printed:
            reclevel -= 1
            if reclevel == 0:
                return '...'
        else:
            printed.add(self)
        kdr = self.cdr
        if isinstance(kdr, Promise):
            kdr = kdr.value()
        if kdr is NIL:
            s = lstr(self.car, print_quote, reclevel, printed)
            return s
        elif isinstance(kdr, Cell):
            s = lstr(self.car, print_quote, reclevel, printed)
            t = kdr._lrepr(print_quote, reclevel, printed)
            return s + ' ' + t
        else:
            s = lstr(self.car, print_quote, reclevel, printed)
            t = lstr(kdr, print_quote, reclevel, printed)
            return s + ' . ' + t


def lstr(x, print_quote=True, reclevel=MAX_EXPANSIONS, printed=None):
    "引数の Lisp 式としての文字列表現"
    if printed is None:
        printed = set()
    if x is NIL:
        return 'nil'
    elif x is True:
        return 't'
    elif isinstance(x, Cell):
        if x.car is S_QUOTE and isinstance(x.cdr, Cell) and x.cdr.cdr is NIL:
            return "'" + lstr(x.cdr.car, print_quote, reclevel, printed)
        else:
            return '(' + x._lrepr(print_quote, reclevel, printed) + ')'
    elif isinstance(x, str):
        if print_quote:
            s = repr(x)
            if s[0] == '"':
                return s
            else:
                return '"' + s.replace('"', '\\"')[1:-1] + '"'
        else:
            return x
    else:
        return str(x)


def llist(*args):
    "Lisp の list 関数に相当"
    z = y = Cell(NIL, NIL)
    for e in args:
        y.cdr = Cell(e, NIL)
        y = y.cdr
    return z.cdr


def mapcar(fn, x):
    "Lisp の mapcar 関数に相当"
    z = y = Cell(NIL, NIL)
    for e in x:
        y.cdr = Cell(fn(e), NIL)
        y = y.cdr
    return z.cdr


# 準引用 (Quasi-Quotation) のためのクラスと関数群

class QQ_Unquote (object):
    __slots__ = 'x'

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return ',' + lstr(self.x)


class QQ_UnquoteSplicing (object):
    __slots__ = 'x'

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return ',@' + lstr(self.x)


def QQ_expand(x):
    "準引用式 `x の x を等価な S 式に展開する"
    if isinstance(x, Cell):
        t = QQ__expand1(x)
        if isinstance(t, Cell) and t.cdr is NIL:
            k = t.car
            if isinstance(k, Cell) and k.car in (S_LIST, S_CONS):
                return k
        return Cell(S_APPEND, t)
    elif isinstance(x, QQ_Unquote):
        return x.x
    else:
        return QQ__quote(x)

def QQ__quote(x):
    if isinstance(x, (Symbol, Arg, Cell)):
        return llist(S_QUOTE, x)
    else:
        return x

def QQ__expand1(x):
    if isinstance(x, Cell):
        h = QQ__expand2(x.car)
        t = QQ__expand1(x.cdr)
        if isinstance(t, Cell):
            if t.car is NIL and t.cdr is NIL:
                return llist(h)
            elif isinstance(h, Cell) and h.car is S_LIST:
                if isinstance(t.car, Cell) and t.car.car is S_LIST:
                    hh = QQ__concat(h, t.car.cdr)
                else:
                    hh = QQ__conscons(h.cdr, t.car)
                return Cell(hh, t.cdr)
        return Cell(h, t)
    elif isinstance(x, QQ_Unquote):
        return llist(x.x)
    else:
        return llist(QQ__quote(x))

def QQ__concat(x, y):
    if x is NIL:
        return y
    else:
        return Cell(x.car, QQ__concat(x.cdr, y))

def QQ__conscons(x, y):
    if x is NIL:
        return y
    else:
        return llist(S_CONS, x.car, QQ__conscons(x.cdr, y))

def QQ__expand2(x):
    if isinstance(x, QQ_Unquote):
        return llist(S_LIST, x.x)
    elif isinstance(x, QQ_UnquoteSplicing):
        return x.x
    else:
        return llist(S_LIST, QQ_expand(x))


class DefinedFunction (object):
    "ラムダ式などの便宜的な基底クラス"
    __slots__ = 'arity', 'body', 'env'

    def __init__(self, arity, body, env=NIL):
        self.arity = arity
        self.body = body
        self.env = env


class MACRO (DefinedFunction):
    __slots__ = ()

    def __str__(self):
        return lstr(Cell(Symbol('#<macro>'), Cell(self.arity, self.body)))
    

class LAMBDA (DefinedFunction):
    __slots__ = ()

    def __str__(self):
        return lstr(Cell(Symbol('#<lambda>'), Cell(self.arity, self.body)))


class CLOSURE (DefinedFunction):
    __slots__ = ()

    def __str__(self):
        return lstr(Cell(Symbol('#<closure'),
                         Cell(Cell(self.arity, self.env), self.body)))


class Arg (object):
    "コンパイル後のラムダ式やマクロ式の仮引数"
    __slots__ = 'level', 'offset', 'symbol'

    def __init__(self, level, offset, symbol):
        self.level = level
        self.offset = offset
        self.symbol = symbol

    def __repr__(self):
        return 'Arg(%r, %r, %r)' % (self.level, self.offset, self.symbol)

    def __str__(self):
        return '#%s:%s:%s' % (self.level, self.offset, self.symbol)

    def set_value(self, x, env):
        "与えられた環境で仮引数に x を代入する"
        for i in range(self.level):
            env = env.cdr
        env.car[self.offset] = x

    def get_value(self, env):
        "与えられた環境で仮引数の値を得る"
        for i in range(self.level):
            env = env.cdr
        return env.car[self.offset]


class Dummy (object):
    "コンパイル後のマクロ式の dummy symbol"
    __slots__ = 'symbol'

    def __init__(self, symbol):
        self.symbol = symbol

    def __repr__(self):
        return "Dummy(%r)" % self.symbol

    def __str__(self):
        return ":%s:%x" % (self.symbol, id(self))


class Promise (object):
    "約束，つまり Lisp 式 (delay exp) の評価結果"
    __slots__ = '__exp', '__env', '__interp'

    def __init__(self, exp, env, interp):
        self.__exp = exp
        self.__env = env        # 環境 (約束をかなえたら None にする)
        self.__interp = interp

    def __repr__(self):
        return 'Promise(%r, %r, %r)' % (self.__exp, self.__env, self.__interp)

    def __str__(self):
        if self.__env is None:
            return lstr(self.__exp)
        else:
            return '#<promise:%x>' % id(self)

    def value(self):
        if self.__env is None:
            return self.__exp
        else:
            return self

    def deliver(self):
        "約束をかなえる"
        if self.__env is not None:
            old_env = self.__interp.environ
            self.__interp.environ = self.__env
            try:
                x = self.__interp.eval(self.__exp, True)
                if isinstance(x, Promise):
                    x = x.deliver()
            finally:
                self.__interp.environ = old_env
            if self.__env is not None: # eval の中でかなえられていなければ
                self.__exp = x
                self.__env = None
        return self.__exp


class Reader (object):
    "Lisp 式を読む"
    __slots__ = '__rf', '__buf', '__line', '__lineno', '__token'

    def __init__(self, rf, lineno=0):
        self.__rf = rf          # 読み取りファイル
        self.__buf = []         # 入力行から得たトークンの並び
        self.__line = None      # 入力行: str または None
        self.__lineno = lineno  # 入力行の行番号

    def read(self):
        try:
            try:
                self.__read_token()
                return self.__parse_expression()
            except SyntaxError:
                ex = sys.exc_info()[1]
                del self.__buf[:] # 行の残りを捨てて次回の回復を図る
                raise EvalError("SyntaxError: %s -- %d: %s" % 
                                (ex, self.__lineno, self.__line))
        finally:
            if isinstance(self.__rf, InteractiveInput):
                self.__rf.reset()

    def __parse_expression(self):
        token = self.__token
        if token is _DOT or token is _RPAREN:
            raise SyntaxError('unexpected %s' % token)
        elif token is _LPAREN:
            self.__read_token()
            return self.__parse_list_body()
        elif token is _SQUOTE:
            self.__read_token()
            return llist(S_QUOTE, self.__parse_expression())
        elif token is _TILDA:
            self.__read_token()
            return llist(S_DELAY, self.__parse_expression())
        elif token is _BQUOTE:
            self.__read_token()
            return QQ_expand(self.__parse_expression())
        elif token is _COMMA:
            self.__read_token()
            return QQ_Unquote(self.__parse_expression())
        elif token is _COMMA_AT:
            self.__read_token()
            return QQ_UnquoteSplicing(self.__parse_expression())
        else:
            return token

    def __parse_list_body(self):
        token = self.__token
        if token is S_EOF:
            raise SyntaxError('unexpected EOF')
        elif token is _RPAREN:
            return NIL
        else:
            e1 = self.__parse_expression()
            self.__read_token()
            if self.__token is _DOT:
                self.__read_token()
                if self.__token is S_EOF:
                    raise SyntaxError('unexpected EOF')
                e2 = self.__parse_expression()
                self.__read_token()
                if self.__token is not _RPAREN:
                    raise SyntaxError('")" expected: %s' % self.__token)
            else:
                e2 = self.__parse_list_body()
            return Cell(e1, e2)

    def __read_token(self):
        while not self.__buf:
            self.__line = self.__rf.readline()
            self.__lineno += 1
            if not self.__line:
                self.__token = S_EOF
                self.__rf.close()
                return
            line = self.__line.strip()
            self.__buf = [t for t in _TOKEN_PAT.findall(line) if t]
        t = self.__buf.pop(0)
        if t in _PARENS_ETC:
            self.__token = Symbol(t)
        elif t == 'nil':
            self.__token = NIL
        elif t == 't':
            self.__token = True
        elif t[0] == t[-1] == '"':
            self.__token = eval(t, {})
        else:
            if t[0] == '-':
                if '0' <= t[1:2] <= '9':
                    try:
                        self.__token = - int(t[1:], 10)
                    except ValueError:
                        pass
                    else:
                        return
            else:
                radix, offset = 10, 0
                if t[0] == '#' and len(t) >= 3:
                    t1 = t[1].lower()
                    if t1 == 'b':
                        radix, offset = 2, 2
                    elif t1 == 'o':
                        radix, offset = 8, 2
                    elif t1 == 'x':
                        radix, offset = 16, 2
                try:
                    self.__token = int(t[offset:], radix)
                except ValueError:
                    pass
                else:
                    return
            try:
                self.__token = float(t)
            except ValueError:
                if _SYMBOL_PAT.match(t):
                    self.__token = Symbol(t)
                else:
                    raise SyntaxError('bad token: %r' % t)


_TOKEN_PAT = re.compile('\\s+|;.*$|(".*?"|,@?|[^()\'`~ ]+|.)')
_PARENS_ETC = ['(', ')', '.', "'", '~', '`', ',', ',@']
_SYMBOL_PAT = re.compile(r'^([A-Za-z0-9]|_|&|\$|\*|/|%|\+|-|<|>|=|!|\?)+$')

_BQUOTE = Symbol('`')
_COMMA = Symbol(',')
_COMMA_AT = Symbol(',@')
_DOT = Symbol('.')
_LPAREN = Symbol('(')
_RPAREN = Symbol(')')
_SQUOTE = Symbol("'")
_TILDA = Symbol('~')


def car(x):
    if x is NIL:
        return NIL
    else:
        return x.car

def cdr(x):
    if x is NIL:
        return NIL
    else:
        return x.cdr

def prin1(x):
    sys.stdout.write(lstr(x, True))
    return x

def princ(x):
    sys.stdout.write(lstr(x, False))
    return x

def terpri():
    sys.stdout.write('\n')
    return True

def _add(*x):
    result = 0
    for e in x:
        result += e
    return result

def _mul(*x):
    result = 1
    for e in x:
        result *= e
    return result

def _div(x, *y):
    for e in y:
        if isinstance(x, int) and isinstance(e, int):
            x //= e
        else:
            x /= e
    return x

def _sub(x, *y):
    if y:
        for e in y:
            x -= e
        return x
    else:
        return -x

def rplaca(x, y):
    x.car = y
    return y

def rplacd(x,  y):
    x.cdr = y
    return y

def throw(x, y):
    raise Thrown(x, y)

def _concat(x):
    if isinstance(x, str):
        return x
    else:
        return ''.join([chr(e) for e in x])

def _print(x):
    print(x)
    return NIL


class Interp (object):
    "Lisp 式を解釈する"
    __slots__ = '__reader', 'environ', 'symbol', 'lazy', 'python_env'

    def __init__(self):
        self.__reader = Reader(InteractiveInput('', ''))
        self.environ = NIL
        self.symbol = {}
        self.lazy = set()
        self.python_env = {}   # python-exec, python-eval のための環境
        self.symbol[S_ERROR] = S_ERROR
        self.__def('*version*', llist(__version__, 'Python'))
        self.__def('*eof*', S_EOF)
        self.__def('car', car)
        self.__def('cdr', cdr)
        self.__def('cons', Cell)
        self.lazy.add(Cell)
        self.__def('atom', lambda x: not isinstance(x, Cell) or NIL)
        self.__def('stringp', lambda x: isinstance(x, str) or NIL)
        self.__def('numberp', lambda x: isinstance(x, _NUMS) or NIL)
        self.__def('eq', lambda x, y: x is y or NIL)
        self.__def('eql', lambda x, y: x == y or NIL)
        self.__def('list', llist)
        self.lazy.add(llist)
        self.__def('prin1', prin1)
        self.__def('princ', princ)
        self.__def('terpri', terpri)
        self.__def('read', self.__reader.read)
        self.__def('+', _add)
        self.__def('*', _mul)
        self.__def('/', _div)
        self.__def('-', _sub)
        self.__def('%', operator.mod)
        self.__def('<', lambda x, y: x < y or NIL)
        self.__def('load', lambda fname: self.run(open(fname)))
        self.__def('eval', self.__eval)
        self.__def('apply', lambda fn, a: self.apply(fn, list(a)))
        self.__def('force', lambda x: x)
        self.__def('rplaca', rplaca)
        self.__def('rplacd', rplacd)
        self.__def('throw', throw)
        self.__def('mapcar', lambda fn, x:
                   mapcar(lambda e: self.apply(fn, [e]), x))
        self.__def('mapc', self.__mapc)
        self.__def('length', len)
        self.__def('_add', operator.add)
        self.__def('_concat', _concat)
        self.__def('python-exec', self.__python_exec)
        self.__def('python-eval', lambda x: eval(x, self.python_env))
        self.__def('python-apply', lambda fn, a, k: fn(*a, **dict(k)))
        self.__def('python-self', lambda: self)
        self.__def('dump', lambda:
                   llist(llist(*self.symbol.keys()), self.environ))
        self.__def('help', lambda: _print(__doc__))
        self.__def('prelude', lambda: _print(PRELUDE))
        self.__def('copyright', lambda: _print(COPYRIGHT))
        self.run(PRELUDE)

    def __def(self, name, fn):
        self.symbol[Symbol(name)] = fn

    def __eval(self, x):
        old_env = self.environ
        self.environ = NIL      # 大域的な環境にする
        try:
            return self.eval(x, True) # 末尾呼出しと同じく環境復元は不要
        finally:
            self.environ = old_env

    def __mapc(self, fn, x):
        for e in x:
            self.apply(fn, [e])
        return x

    def __python_exec(self, x):
        exec(x, self.python_env)
        return NIL

    def eval(self, x, can_lose_current_env=False):
        try:
            while True:
                if isinstance(x, Symbol):
                    return self.__eval_symbol(x)
                elif isinstance(x, Arg):
                    return x.get_value(self.environ)
                elif isinstance(x, Cell):
                    kar = x.car
                    if kar is S_QUOTE:
                        return x.cdr.car
                    elif kar is S_PROGN:
                        x = self.__eval_progn_body(x.cdr)
                    elif kar is S_COND:
                        (x, cont) = self.__eval_cond_body(x.cdr)
                        if not cont:
                            return x
                    elif kar is S_SETQ:
                        return self.__eval_setq_body(x.cdr)
                    elif kar is S_LAMBDA:
                        return CLOSURE(env=self.environ,
                                       *self.__compile(x.cdr))
                    elif kar is S_MACRO:
                        if self.environ is not NIL:
                            raise EvalError('nested macro', x)
                        y = _replace_dummy_variables(x.cdr, {})
                        return MACRO(*self.__compile(y))
                    elif kar is S_CATCH:
                        return self.__eval_catch_body(x.cdr)
                    elif kar is S_UNWIND_PROTECT:
                        return self.__eval_unwind_protect_body(x.cdr)
                    elif kar is S_DELAY:
                        kdr = x.cdr
                        if not (isinstance(kdr, Cell) and kdr.cdr is NIL):
                            raise EvalError('bad delay')
                        return Promise(kdr.car, self.environ, self)
                    else:
                        fn = x.car
                        # 高速化のため eval をここに簡単に展開する
                        if isinstance(fn, Symbol):
                            fn = self.__eval_symbol(fn)
                        elif isinstance(fn, Arg):
                            fn = fn.get_value(self.environ)
                        elif isinstance(fn, Cell):
                            fn = self.eval(fn)
                        elif isinstance(fn, LAMBDA):
                            fn = CLOSURE(fn.arity, fn.body, self.environ)
                        #
                        if isinstance(fn, Promise):
                            fn = fn.deliver()
                        #
                        if isinstance(fn, CLOSURE):
                            args = self.__get_args(x.cdr, False)
                            (x, cont) = self.__apply_function(
                                fn, args, can_lose_current_env)
                            if not cont:
                                return x
                        elif isinstance(fn, MACRO):
                            (x, cont) = self.__apply_function(fn, list(x.cdr))
                        else:
                            args = self.__get_args(x.cdr, fn not in self.lazy)
                            try:
                                return fn(*args)
                            except (EvalError, KeyboardInterrupt, SystemExit):
                                raise
                            except:
                                info = sys.exc_info()
                                raise EvalError('%s: %s -- %s %s' % (
                                        info[0], info[1], fn, args))
                elif isinstance(x, LAMBDA):
                    return CLOSURE(x.arity, x.body, self.environ)
                else:
                    return x
        except EvalError:
            ex = sys.exc_info()[1]
            if len(ex.trace) < MAX_EXC_TRACES:
                ex.trace.append(lstr(x))
            raise

    def apply(self, fn, args):
        "関数適用のための便宜メソッド"
        if isinstance(fn, (CLOSURE, MACRO)):
            return self.__apply_function(fn, args)[0]
        else:
            if fn not in self.lazy:
                for i, e in enumerate(args):
                    if isinstance(e, Promise):
                        args[i] = e.deliver() # NB args 自体を書き換える
            return fn(*args)

    def __eval_symbol(self, name):
        try:
            return self.symbol[name]
        except KeyError:
            raise EvalError('void variable', name)

    def __eval_progn_body(self, body):
        if isinstance(body, Cell):
            d = body.cdr
            while isinstance(d, Cell):
                self.eval(body.car)
                body, d = d, d.cdr
            if d is not NIL:
                raise ProperListExpected(d)
            return body.car     # 末尾呼出し ⇒ 戻った先で評価
        else:
            if body is not NIL:
                raise ProperListExpected(body)
            return NIL

    def __eval_cond_body(self, body):
        while isinstance(body, Cell):
            clause = body.car
            if isinstance(clause, Cell):
                result = self.eval(clause.car)
                if isinstance(result, Promise):
                    result = result.deliver()
                if result is not NIL: # テスト結果が真ならば
                    clause = clause.cdr
                    if not isinstance(clause, Cell):
                        return (result, False)
                    d = clause.cdr
                    while isinstance(d, Cell):
                        self.eval(clause.car)
                        clause, d = d, d.cdr
                    if d is not NIL:
                        raise ProperListExpected(d)
                    return (clause.car, True) # 末尾呼出し ⇒ 戻った先で評価
            elif clause is not NIL:
                raise EvalError('cond test expected', clause)
            body = body.cdr
        if body is not NIL:
            raise ProperListExpected(body)
        return (NIL, False)     # すべて失敗ならば nil

    def __eval_setq_body(self, body): # (LVAL RVAL LVAL RVAL...)
        result = NIL
        while isinstance(body, Cell):
            lval, body = body.car, body.cdr
            if not isinstance(body, Cell):
                raise EvalError('right value expected')
            result = self.eval(body.car)
            if isinstance(lval, Symbol):
                self.symbol[lval] = result
            elif isinstance(lval, Arg):
                lval.set_value(result, self.environ)
            else:
                raise VariableExpected(lval)
            body = body.cdr
        if body is not NIL:
            raise ProperListExpected(body)
        return result

    def __get_args(self, aa, flag):
        """Lisp のリスト aa を評価して Python のリストを得る。
        flag が真ならば，評価した後さらに force する。
        """
        args = []
        while isinstance(aa, Cell):
            x = self.eval(aa.car)
            if flag and isinstance(x, Promise):
                x = x.deliver()
            args.append(x)
            aa = aa.cdr
        if aa is not NIL:
            raise ProperListExpected(aa)
        return args

    def __apply_function(self, fn, args, can_lose_original_env=False):
        assert isinstance(fn, (CLOSURE, MACRO))
        body = fn.body
        if not isinstance(body, Cell):
            raise EvalError('body expected')
        arity, env = fn.arity, fn.env
        if arity < 0:           # &rest 付きならば
            arity = -arity - 1  # &rest より前の引数の個数を得て
            if arity <= len(args): # rest 引数を１個の Lisp リストに構成する
                args, rest = args[:arity], args[arity:]
                args.append(llist(*rest))
                arity += 1
        if len(args) != arity:
            raise EvalError('arity not matched')
        old_env = self.environ          # 元の環境を退避する
        self.environ = Cell(args, env)  # 新環境に変更する
        try:
            d = body.cdr
            while isinstance(d, Cell):
                self.eval(body.car)
                body, d = d, d.cdr
            if can_lose_original_env:   # ⇒ 典型的には末尾呼出し
                old_env = self.environ  # 新環境のまま
                return (body.car, True) # 戻った先で評価する
            else:
                return (self.eval(body.car, True), False)
        finally:
            self.environ = old_env

    def __compile(self, j):
        if not isinstance(j, Cell):
            raise EvalError('arglist and body expected')
        (has_rest, table) = _make_arg_table(j.car)
        arity = len(table)
        if has_rest:
            arity = -arity
        if not isinstance(j.cdr, Cell):
            raise EvalError('body expected')
        body = _scan(j.cdr, table)
        body = self.__expand_macros(body, MAX_MACRO_EXPS)
        body = self.__compile_inners(body)
        return (arity, body)

    def __compile_inners(self, j):
        if isinstance(j, Cell):
            kar = j.car
            if kar is S_QUOTE:
                return j
            elif kar is S_LAMBDA:
                return LAMBDA(*self.__compile(j.cdr))
            elif kar is S_MACRO:
                raise EvalError('nested macro', j)
            else:
                return mapcar(self.__compile_inners, j)
        else:
            return j

    def __expand_macros(self, j, count):
        def expand(j):
            if count > 0 and isinstance(j, Cell):
                k = j.car
                if k is S_QUOTE or k is S_LAMBDA or k is S_MACRO:
                    return j
                else:
                    if isinstance(k, Symbol):
                        k = self.symbol.get(k, k)
                    if isinstance(k, MACRO):
                        (z, cont) = self.__apply_function(k, list(j.cdr))
                        return self.__expand_macros(z, count - 1)
                    else:
                        return mapcar(expand, j)
            else:
                return j
        return expand(j)

    def __eval_catch_body(self, j): # j = (tag body...)
        if not isinstance(j, Cell):
            raise EvalError('tag and boy expected', j)
        tag = self.eval(j.car)
        try:
            result = NIL
            k = j.cdr
            if isinstance(k, Cell):
                for x in k:
                    result = self.eval(x)
            elif k is not NIL:
                raise ProperListExpected(k)
            return result
        except Thrown:
            th = sys.exc_info()[1]
            if tag == th.tag:
                return th.value
            else:
                raise
        except EvalError:       # 一般の評価時例外の捕捉
            if tag is S_ERROR:
                return sys.exc_info()[1]
            else:
                raise

    def __eval_unwind_protect_body(self, j): # j = (body cleanup...)
        if not isinstance(j, Cell):
            raise EvalError('body (and cleanup) expected', j)
        try:
            return self.eval(j.car)
        finally:
            k = j.cdr
            if isinstance(k, Cell):
                for x in k:
                    self.eval(x)
            elif k is not NIL:
                raise ProperListExpected(k)

    def run(self, rf=None):
        """引数のファイルまたは文字列から式の並びを読んで評価する。
        無引数ならば対話的に入力/評価/出力を繰り返す"""
        interactive = rf is None
        if interactive:
            rf = InteractiveInput('> ', '  ')
        elif isinstance(rf, str):
            rf = StringIO(rf)
        rr = Reader(rf)
        result = NIL
        while True:
            try:
                x = rr.read()
                if x is S_EOF:
                    if interactive:
                        print('Goodbye')
                    return result
                result = self.eval(x)
                if interactive:
                    print(lstr(result))
            except KeyboardInterrupt:
                if interactive:
                    print('\n' 'KeyboardInterrupt')
                else:
                    raise
            except EvalError:
                if interactive:
                    print(sys.exc_info()[1])
                else:
                    raise


class InteractiveInput (object):
    "プロンプト付き標準入力 (可能ならば行編集機能付き)"
    __slots__ = '__ps1', '__ps2', '__primary'

    def __init__(self, ps1, ps2):
        "引数は１次プロンプトと２次プロンプト"
        self.__ps1 = ps1
        self.__ps2 = ps2
        self.__primary = True

    def readline(self):
        "初回は１次プロンプト，次回以降は２次プロンプトで１行を入力する"
        try:
            if self.__primary:
                prompt = self.__ps1
                self.__primary = False
            else:
                prompt = self.__ps2
            return raw_input(prompt) + '\n'
        except EOFError:
            return ''

    def reset(self):
        "プロンプトを１次に戻す"
        self.__primary = True

    def close(self):
        pass


def _replace_dummy_variables(j, names):
    "$ ではじまるシンボルを Dummy に置き換える"
    def replace(j):
        if isinstance(j, Symbol):
            if str(j)[0] == '$':
                k = names.get(j)
                if k is None:
                    names[j] = k = Dummy(j)
                return k
            else:
                return j
        elif isinstance(j, Cell):
            return mapcar(replace, j)
        else:
            return j
    return replace(j)

def _make_arg_table(i):
    "仮引数並び -> (has_rest, table)"
    offset = 0
    has_rest = False
    table = {}
    while isinstance(i, Cell):
        j = i.car
        if has_rest:
            raise EvalError('2nd rest', j)
        if j is S_REST: # &rest rest_arg
            i = i.cdr
            if not isinstance(i, Cell):
                raise VariableExpected(i)
            j = i.car
            if j == S_REST:
                raise VariableExpected(j)
            has_rest = True
        if isinstance(j, Symbol):
            sym = j
        elif isinstance(j, Arg):
            sym = j = j.symbol
        elif isinstance(j, Dummy):
            sym = j.symbol
        else:
            raise VariableExpected(j)
        table[j] = Arg(0, offset, sym)
        offset += 1
        i = i.cdr
    if i is not NIL:
        raise ProperListExpected(i)
    return (has_rest, table)

def _scan(j, table):
    def scan(j):
        if isinstance(j, (Symbol, Dummy)):
            return table.get(j, j)
        elif isinstance(j, Arg):
            k = table.get(j.symbol)
            if k is None:
                return Arg(j.level + 1, j.offset, j.symbol)
            else:
                return k
        elif isinstance(j, Cell):
            if j.car is S_QUOTE:
                return j
            else:
                return mapcar(scan, j)
        else:
            return j
    return scan(j)


# 初期化 Lisp スクリプト
PRELUDE = r'''
(setq defmacro
      (macro (name args &rest body)
             `(progn (setq ,name (macro ,args ,@body))
                     ',name)))

(defmacro defun (name args &rest body)
  `(progn (setq ,name (lambda ,args ,@body))
          ',name))

(defun caar (x) (car (car x)))
(defun cadr (x) (car (cdr x)))
(defun cdar (x) (cdr (car x)))
(defun cddr (x) (cdr (cdr x)))
(defun caaar (x) (car (car (car x))))
(defun caadr (x) (car (car (cdr x))))
(defun cadar (x) (car (cdr (car x))))
(defun caddr (x) (car (cdr (cdr x))))
(defun cdaar (x) (cdr (car (car x))))
(defun cdadr (x) (cdr (car (cdr x))))
(defun cddar (x) (cdr (cdr (car x))))
(defun cdddr (x) (cdr (cdr (cdr x))))
(defun not (x) (eq x nil))
(defun consp (x) (not (atom x)))
(defun print (x) (prin1 x) (terpri) x)
(defun identity (x) x)

(setq
 = eql
 null not
 setcar rplaca
 setcdr rplacd)

(defun > (x y) (< y x))
(defun >= (x y) (not (< x y)))
(defun <= (x y) (not (< y x)))
(defun /= (x y) (not (= x y)))

(defun equal (x y)
  (cond ((atom x) (eql x y))
        ((atom y) nil)
        ((equal (car x) (car y)) (equal (cdr x) (cdr y)))))

(defun concat (&rest x)
  (cond ((null x) "")
        ((null (cdr x)) (_concat (car x)))
        (t (_add (_concat (car x))
                 (apply concat (cdr x))))))

(defmacro if (test then &rest else)
  `(cond (,test ,then)
         ,@(cond (else `((t ,@else))))))

(defmacro when (test &rest body)
  `(cond (,test ,@body)))

(defmacro let (args &rest body)
  ((lambda (vars vals)
     (defun vars (x)
       (cond (x (cons (if (atom (car x))
                          (car x)
                        (caar x))
                      (vars (cdr x))))))
     (defun vals (x)
       (cond (x (cons (if (atom (car x))
                          nil
                        (cadar x))
                      (vals (cdr x))))))
     `((lambda ,(vars args) ,@body) ,@(vals args)))
   nil nil))

(defun _append (x y)
  (if (null x)
      y
    (cons (car x) (_append (cdr x) y))))
(defmacro append (x &rest y)
  (if (null y)
      x
    `(_append ,x (append ,@y))))

(defmacro and (x &rest y)
  (if (null y)
      x
    `(cond (,x (and ,@y)))))

(defmacro or (x &rest y)
  (if (null y)
      x
    `(cond (,x)
           ((or ,@y)))))

(defun listp (x)
  (or (null x) (consp x)))    ; NB (listp (lambda (x) (+ x 1))) => nil

(defun memq (key x)
  (cond ((null x) nil)
        ((eq key (car x)) x)
        (t (memq key (cdr x)))))

(defun member (key x)
  (cond ((null x) nil)
        ((equal key (car x)) x)
        (t (member key (cdr x)))))

(defun assq (key alist)
  (cond (alist (let ((e (car alist)))
                 (if (and (consp e) (eq key (car e)))
                     e
                   (assq key (cdr alist)))))))

(defun assoc (key alist)
  (cond (alist (let ((e (car alist)))
                 (if (and (consp e) (equal key (car e)))
                     e
                   (assoc key (cdr alist)))))))

(defun _nreverse (x prev)
  (let ((next (cdr x)))
    (setcdr x prev)
    (if (null next)
        x
      (_nreverse next x))))
(defun nreverse (list)            ; (nreverse '(a b c d)) => (d c b a)
  (cond (list (_nreverse list nil))))

(defun last (list)
  (if (atom (cdr list))
      list
    (last (cdr list))))

(defun nconc (&rest lists)
  (if (null (cdr lists))
      (car lists)
    (setcdr (last (car lists))
            (apply nconc (cdr lists)))
    (car lists)))

(defmacro push (newelt listname)
  `(setq ,listname (cons ,newelt ,listname)))

(defmacro pop (listname)
  `(let (($a (car ,listname)))
     (setq ,listname (cdr ,listname))
     $a))

(defmacro while (test &rest body)
  `(let ($loop)
     (setq $loop (lambda () (cond (,test ,@body ($loop)))))
     ($loop)))

(defun nth (n list)
  (while (< 0 n)
    (setq list (cdr list)
          n (- n 1)))
  (car list))

(defmacro dolist (spec &rest body) ; (dolist (name list [result]) body...)
  (let ((name (car spec)))
    `(let (,name
           ($list ,(cadr spec)))
       (while $list
         (setq ,name (car $list))
         ,@body
         (setq $list (cdr $list)))
       ,@(if (cddr spec)
             `((setq ,name nil)
               ,(caddr spec))))))

(defmacro dotimes (spec &rest body) ; (dotimes (name count [result]) body...)
  (let ((name (car spec)))
    `(let ((,name 0)
           ($count ,(cadr spec)))
       (while (< ,name $count)
         ,@body
         (setq ,name (+ ,name 1)))
       ,@(if (cddr spec)
             `(,(caddr spec))))))

(defun reduce (f x)
  (if (null x)
      (f)
    (let ((r (car x)))
      (setq x (cdr x))
      (while x
        (setq r (f r (car x))
              x (cdr x)))
      r)))

(defun some (f x)
  (cond ((null x) nil)
        ((f (car x)))
        (t (some f (cdr x)))))

(defun take (n x)                       ; Haskell
  (if (or (= 0 n) (null x))
      nil
    (cons (car x) (take (- n 1) (cdr x)))))

(defun drop (n x)                       ; Haskell
  (if (or (= 0 n) (null x))
      x
    (drop (- n 1) (cdr x))))

(defun _zip (x)
  (if (some null x)
      nil
    (let ((cars (mapcar car x))
          (cdrs (mapcar cdr x)))
      (cons cars ~(_zip cdrs)))))
(defun zip (&rest x) (_zip x))          ; Python 3.0 & Haskell

(defun range (m n)                      ; Python 3.0
  (cond ((< m n) (cons m ~(range (+ m 1) n)))))

(defun map (f x)                        ; Haskell
  (cond (x (cons ~(f (car x)) ~(map f (cdr x))))))

(defun mapf (f x)                       ; map force
  (cond (x (cons (f (car x)) ~(map f (cdr x))))))

(defun scanl (f q x)                    ; Haskell
  (cons q ~(cond (x (scanl f (f q (car x)) (cdr x))))))

(defun filter (f x)                     ; Haskell & Python 3.0
  (cond ((null x) nil)
        ((f (car x)) (cons (car x) ~(filter f (cdr x))))
        (t (filter f (cdr x)))))

(python-exec
 (concat
  "import sys\n"
  "def printf(fs, *args):\n"
  "  sys.stdout.write(fs % args)\n"))
(setq _printf (python-eval "printf"))
(defun printf (fs &rest args)
  (python-apply _printf (cons fs args) nil))

(defun exit (n)
  (python-apply (python-eval "sys.exit") (list n) nil))
'''


if __name__ == '__main__':      # 主ルーチン
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    interp = Interp()
    file_names = sys.argv[1:]
    if file_names:
        for name in file_names:
            if name == '-':
                interp.run()
            else:
                interp.run(open(name))
    else:
        interp.run()
