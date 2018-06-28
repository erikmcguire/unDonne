import tkinter as tk
from tkinter import *
from main import *

impdflt = "Export path."
tdflt = "Sonnet seed theme (char-OMM only, single word)"
vmdflt = "Vector model (char-OMM only)."

class GenGUI():
    def __init__(self):
        self._root = tk.Tk()
        self._root.title("Sonnet Generator")
        self._tex = tk.Text()
        self._v = IntVar()
        self._bframe=Frame(self._root)
        self._pframe=Frame(self._root)
        self._tframe=Frame(self._root)
        self._imp = tk.Entry(master=self._bframe, width=len(impdflt) + 2)
        self._topic = tk.Entry(master=self._bframe, width=len(tdflt) + 1)
        self._vmod = tk.Entry(master=self._tframe,
                              width=len(vmdflt) + 1)
        self._vm = None

    def _get_out(self, t, sonnett, path="./output/"):
        """Export to .txt."""
        inpath = self._imp.get()
        path = inpath if inpath != "Export path." else path
        fpath = "{}{}_{}{}".format(path, t, time.time(), ".txt")
        with open(fpath, 'w') as f:
            f.write(sonnett)

    def _genfn(self, t, vm):
        titlek = None
        if t == 'char-OMM':
            ntheme = self._topic.get()
            ctheme = ntheme if ntheme.isalpha() and len(ntheme.split()) == 1 else alpha
            karkoav = Karkoav(f="data/sonnets.txt",
                              min=min_chars(hslines),
                              max=max_chars(hslines),
                              hsl = hslines,
                              k=5, sub1=0.06, sub2=0.03,
                              p=kpunk,
                              theme=ctheme,
                              thresh=100,
                              vmod=self._vm if self._vm != vmdflt else "fasttext",
                              default = True)
            sonnett, titlek = karkoav.generate()
        else:
            if t == 'PCFG':
                sonnett = gen_pcfg(kpunk, d=8)
            elif t == 'Bigrams':
                sonnett = gen_from_cfd(kpunk, cpd,
                                       seed, hslines)
        lbl = tk.Label(master=self._bframe,
                       text="Generated {} sonnet: ".format(t))
        lbl.pack(side=TOP)
        self._tex.configure(font=('Garamond', 14),
                            background="antiquewhite")
        self._tex.insert(1.0, "\n{}".format(titlek + '\n\n' + sonnett if titlek else sonnett))
        e_btn = tk.Button(master=self._tframe,
                          text="Export",
                          command=lambda:
                                    self._get_out(titlek or t, sonnett))
        e_btn.pack()
        self._tex.pack(expand=True, fill=BOTH)
        if t == 'char-OMM' and self._v.get() == 1:
            self._imp.focus_set()
            dd, title, lines = karkoav.get_most_contrib()
            karkoav.gen_svg(dd=dd, title=re.sub(r"[^A-z\W]", "", title), lines=lines)

    def _make_button(self, t):
        if t == 'c':
            btn = Checkbutton(master=self._bframe,
                              text="Visualize (char-OMM only)",
                              variable=self._v)
            btn.pack(anchor=W)
        else:
            btn = tk.Button(master=self._bframe,
                            text="Generate w/ {}".format(t),
                            command=lambda: self._genfn(t, self._vm))
            btn.pack(side=LEFT)


    def main(self):
        self._bframe.pack()
        self._pframe.pack(side=RIGHT)
        self._tframe.pack(side=BOTTOM)

        self._vmod.insert(0, vmdflt)
        self._vmod.pack(side=BOTTOM)
        self._vm = self._vmod.get()

        list(self._make_button(i)
            for i in ['c', 'char-OMM', 'Bigrams', 'PCFG'])

        self._imp.insert(0, impdflt)
        self._topic.insert(0, tdflt)
        self._topic.pack(side=TOP)
        self._imp.pack(side=BOTTOM)
        self._topic.focus_set()

        x = (self._root.winfo_screenwidth() -
             self._root.winfo_reqwidth()) * .4
        y = (self._root.winfo_screenheight() -
             self._root.winfo_reqheight()) * .3
        self._root.geometry("{}x{}+{}+{}".format(650, 650,
                                                 int(x), int(y)))
        self._root.mainloop()
