% mqthesis.cls - Macquarie University PhD Thesis LaTeX Class File.

% Copyright (C) 2004 Paul Cochrane
% Copyright (C) 2008 Alexei Gilchrist
% Modified 2010 Johann-Heinrich Sch\"onfeldt 
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%   
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%    
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mqthesis}[11/02/2010 Macquarie University thesis LaTeX class]

% ifthenelse for if loops
\RequirePackage{ifthen}

% is the thesis for Phd or Honours?  Default is PhD
\newcommand{\degreetext}{for the degree of Doctor of Philosophy}
\DeclareOption{phd}{\renewcommand{\degreetext}%
{for the degree of \\ Doctor of Philosophy}}
\DeclareOption{honours}{\renewcommand{\degreetext}%
{in partial fulfilment of the degree of \\ Bachelor of Science with Honours}}

% print a copyright page on the back of the title page?
\newboolean{copyrightpage}
\setboolean{copyrightpage}{false}
\DeclareOption{copyrightpage}{\setboolean{copyrightpage}{true}}

% print ``Examiner's copy'' on the title page?
\newboolean{examinerscopy}
\setboolean{examinerscopy}{false}
\DeclareOption{examinerscopy}{\setboolean{examinerscopy}{true}}

% use small caps for the title text?
\newboolean{titlesmallcaps}
\setboolean{titlesmallcaps}{false}
\DeclareOption{titlesmallcaps}{\setboolean{titlesmallcaps}{true}}

 % use fancyschmancy hyperref coloured links, or nice'n black for printing?
 \newboolean{foronline}
 \setboolean{foronline}{false}
 \DeclareOption{foronline}{\setboolean{foronline}{true}}

% finished processing options
\ProcessOptions\relax

% load the book class using a4paper
\LoadClass[12pt,a4paper]{book}

% give the header a bit more room for fancyhdr below
% otherwise LaTeX will spew on each page
\addtolength{\headheight}{2.5pt}

% first set to zero ... 
\setlength{\oddsidemargin}{-1in}
\setlength{\evensidemargin}{-1in}
\setlength{\topmargin}{-1in}       

% adjust these if printer is off by a bit
\setlength{\hoffset}{0mm}
\setlength{\voffset}{0mm}

% from HDR Thesis Preparation Advice 2008
% margins >= 3.5cm on binding edge and >= 1.5cm on opposite
%         >= 2.0cm on top and bottom 

% NB also that the optimal number of characters per line 
% for readability is only 60-70, we're over so we'll be a
% bit more generous on the evensidemargin

\addtolength{\oddsidemargin}{35mm} 
\addtolength{\evensidemargin}{20mm}
\addtolength{\topmargin}{20mm}

% set up some of the spacing
\setlength{\marginparwidth}{40pt}  
\setlength{\marginparsep}{10pt}
\setlength{\headsep}{0.5in}

% A4 dimensions [mm]: 209.903 x 297.039
\setlength{\textwidth}{21 cm}
\setlength{\textheight}{29.7 cm}

% fix up width
\addtolength{\textwidth}{-\oddsidemargin}
\addtolength{\textwidth}{-\evensidemargin}
% now we've added 2inches in setting up margins
\addtolength{\textwidth}{-2in}

% fix up height
\addtolength{\textheight}{-2\topmargin}
\addtolength{\textheight}{-\headheight}
\addtolength{\textheight}{-\headsep}
\addtolength{\textheight}{-\footskip}
% now we've added 2inches in setting up margins
\addtolength{\textheight}{-2in}

\brokenpenalty=10000   % dunno what this does, maybe handy



% this stops one figure taking up a whole page and lets more text onto
% the one page when a figure exists
\renewcommand\floatpagefraction{0.8} %   Default = 0.5

% macros for defining the department, various dates and the title and
% author of the thesis.
\def\department#1{\gdef\@department{#1}}
\def\submitdate#1{\gdef\@submitdate{#1}}
\def\copyrightyear#1{\gdef\@copyrightyear{#1}} % \author, \title in report
\def\@title{}
\def\@author{}
\def\@submitdate{\ifcase\the\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\the\year}
\def\@copyrightyear{\number\the\year}

%%% load the required packages
% fancyhdr for nice, fancy headings
\RequirePackage{fancyhdr}
% ccaption for good caption handling
\RequirePackage{ccaption}
% xspace so that spaces after commands are handled correctly
\RequirePackage{xspace}
 % required for nice pictures
\RequirePackage{graphicx} 
% required to use \ifpdf statements, see end of doc
\RequirePackage{ifpdf}

% This constructs the MQ title page
\def\titlep{%
    \thispagestyle{empty}%
    \null\vfill
    \begin{center}
        \ifthenelse{\boolean{titlesmallcaps}}
            {\huge\sc\expandafter{\@title}} % put in small caps
            {\Large\uppercase\expandafter{\@title}} % use standard title
    \end{center}
    \vskip1cm%
    \begin{center}
        \textrm{By}\\
        \vskip1cm%
        {\Large\textrm{\@author}}\\
    \end{center}
    \vfill
    \begin{center}
	\ifthenelse{\boolean{foronline}}{
           \textsc{A thesis submitted to \href{http://www.mq.edu.au/}{Macquarie University}\\
           \degreetext{}\\
           Department of \expandafter{\@department}\\
           \@submitdate\\}
	}{\textsc{A thesis submitted to Macquarie University}\\
           \degreetext{}\\
           Department of \expandafter{\@department}\\
           \@submitdate\\}
        \vspace{1cm}
        \includegraphics[width=0.25 \columnwidth]{UNIlogos/MQlogo}\\
    \end{center}


    \ifthenelse{\boolean{examinerscopy}}{%
        \vspace{1cm}
        \begin{center}
            {\LARGE \textsc{Examiner's Copy}}
        \end{center}%
        }{}
    \vfill
    \newpage}

% this defines the signature for the signature page where the standard
% declaration is put
\long\def\signature#1{%
\begin{center}
\begin{minipage}{4in}
\parindent=0pt
Except where acknowledged in the customary manner, the material
presented in this thesis is, to the best of my knowledge, original and
has not been submitted in whole or part for a degree in any university.
\par
\vspace{.5in}
\hbox to 4in{\hfil\shortstack{\vrule width 3in height 0.4pt\\#1}}
\end{minipage}
\end{center}}

% this is the actual signature page
\def\signaturepage{%
	\vskip.5in
	\signature{\@author}}

% now put in the titlepage
% (bit of a misnomer really, as it is really the first three or four pages.
\def\titlepage{%
    \pagenumbering{roman}  % make sure that front matter is numbered roman
    \ifthenelse{\boolean{foronline}}{\hypersetup{urlcolor=black}}{}	
    \titlep
    \ifthenelse{\boolean{copyrightpage}}{%
        \newpage
        \vspace*{\fill}
        \begin{center}
            \copyright{} \@author, \@copyrightyear.
        \end{center}
        \begin{center}
            Typeset in \LaTeXe{}.
        \end{center}
        \normalsize
        \newpage
        \signaturepage}{}
	\ifthenelse{\boolean{foronline}}{\hypersetup{urlcolor=blue}}{}% url link in document will be blue
}

% improved version of caption handling
% \usepackage{ccaption}
\captionnamefont{\scshape}
\captionstyle{}
\makeatletter
\renewcommand{\fnum@figure}[1]{\quad\small\textsc{\figurename~\thefigure}:}
\renewcommand{\fnum@table}[1]{\quad\small\textsc{\tablename~\thetable}:}
\renewcommand{\@makecaption}[2]{%
\vskip\abovecaptionskip
\sbox\@tempboxa{#1: #2}%
\ifdim \wd\@tempboxa >\hsize
  \def\baselinestretch{1}\@normalsize
  #1: #2\par
  \def\baselinestretch{1.5}\@normalsize
\else
  \global \@minipagefalse
  \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
\fi
\vskip\belowcaptionskip}
\makeatother

% set the pagestyle to look good
\pagestyle{fancy}

%%%%% Fancyhdr stuff
% define how headers are marked, for details, see fancyhdr docs
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}

% define where sections, chapters and pagenumbers are put
% see fancyhdr docs for details
% the \nouppercase stops book.cls making the contents, bibliography
% and index headers from being all in uppercase.
% The options used here are essentially that in Lamport's book, but
% with small caps for the headings.
\fancyhf{}
\fancyhead[LE,RO]{\nouppercase{\thepage}}
\fancyhead[LO]{\sc \nouppercase{\rightmark}}
\fancyhead[RE]{\sc \nouppercase{\leftmark}}

%%% other settings required for a thesis
% It's a references section, not a bibliography, hence redefine
% \bibname i.e. change ``Bibliography'' to ``References''
\renewcommand*{\bibname}{References}

% use 1.5 line spacing for the examiners copy
% single line spacing for the final copy
\ifthenelse{\boolean{examinerscopy}}{%
\renewcommand{\baselinestretch}{1.5}}%
{\renewcommand{\baselinestretch}{1}}

% spell things correctly
\newenvironment{centre}{\begin{center}}{\end{center}}
\newenvironment{itemise}{\begin{itemize}}{\end{itemize}}

%\usepackage{play}
\usepackage[grey,times]{quotchap} % this makes the chapter title look nice, 
                                                        % and you can insert a quote
\usepackage{makeidx}

%%%%% set up the bibliography style
\bibliographystyle{mqthesis}  % mqthesis bibliography style file, made
			                  % with makebst

%%%%% optional packages
%\usepackage[square,comma,numbers,sort&compress]{natbib}
		% this is the natural sciences bibliography citation
		% style package.  The options here give citations in
		% the text as numbers in square brackets, separated by
		% commas, citations sorted and consecutive citations
		% compressed 
		% output example: [1,4,12-15]

\usepackage[nottoc]{tocbibind}  
				% allows the table of contents, bibliography
				% and index to be added to the table of
				% contents if desired, the option used
				% here specifies that the table of
				% contents is not to be added.
				% tocbibind needs to be after natbib
				% otherwise bits of it get trampled.


\ifpdf
%-for more info checkout: http://en.wikibooks.org/wiki/LaTeX/Hyperlinks
% where you can also find the manual with more options.
	\ifthenelse{\boolean{foronline}}{
          \usepackage[pdftex, plainpages = false, pdfpagelabels, 
                 pdfpagelayout = OneColumn,
                 bookmarks,
                 bookmarksopen = true,
                 bookmarksnumbered = true,
                 breaklinks = true,
                 linktocpage,
                 hyperindex = true,
                 hyperfigures,
		pdfdisplaydoctitle = true,
                pagebackref,
                colorlinks=true,
                linkcolor=blue,
                urlcolor=blue,
                citecolor=red,
                anchorcolor=green
                 ]{hyperref}}{\usepackage[pdftex]{hyperref}}
    	\pdfcompresslevel=9
\else
      \ifthenelse{\boolean{foronline}}{
        \usepackage[ dvips, 
                 bookmarks,
                 bookmarksopen = true,
                 bookmarksnumbered = true,
                 breaklinks = true,
                 linktocpage,
                 hyperindex = false,
                 hyperfigures,
                 pagebackref,
                 colorlinks=true,
                 linkcolor=blue,
                 urlcolor=blue,
                 citecolor=red,
                 anchorcolor=green
                 ]{hyperref}}{\usepackage[dvips]{hyperref}}	
\fi

% put in an index?
% \makeindex
