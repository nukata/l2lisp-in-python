# An experimental Lisp interpreter in Python 2 & 3

This is an experimental Lisp interpreter I wrote 10 years ago (2008) in Python.
It had been presented under the MIT License at <http://www.oki-osk.jp/esc/llsp/>
until last spring (2017), which has been shut down now.

## How to use

It runs in any Python from 2.3 to 3.6.

```
$ python L2Lisp.py
> "hello, world"
"hello, world"
> (+ 5 6)
11
> (exit 0)
$
```

You can give it a file name of your Lisp script.

```
$ python L2Lisp.py fibs.l
5702887
$
```

If you put a "`-`" after the file name, the interpreter will 
begin an interactive session after running the file.

```
$ python L2Lisp.py fibs.l -
5702887
> (take 20 fibs)
(1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181 6765)
> (exit 0)
$ 
```

## Features

 - A sort of subset of Emacs Lisp, but being _Lisp-1_ with _lexical scoping_

 - Tail call optimization

 - Automatic avoidance of free symbol capture in macro expansion,
   which makes traditional macros fairly _hygienic_

 - Very few built-ins:  even `defun` is defined in the prelude as follows:
   ```Lisp
   (defmacro defun (name args &rest body)
     `(progn (setq ,name (lambda ,args ,@body))
             ',name))
   ```
 - Implicit forcing for _lazy evaluation_: 
   the Fibonacci sequence can be defined as follows:
   ```Lisp
   (defun zipWith (f x y)
     (if (or (null x)
             (null y)) nil
       (cons (f (car x) (car y))
             ~(zipWith f (cdr x) (cdr y)))))
   
   (setq fibs
         (cons 1 (cons 1 ~(zipWith + fibs (cdr fibs)))))
   ```



## License

This is under the MIT License.
See the [L2Lisp.py](L2Lisp.py#L96-L117) file or 
evaluate `(copyright)` in the Lisp.
