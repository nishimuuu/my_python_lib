#-*-coding: utf-8 -*-
__author__ = '01010357'
import CaboCha
from mecab import MeCabSeparator
from mecab import MeCabToken

# write code here...

class My_cabocha:
  def __init__(self,description):
    self.set_description(description)

  def get_description(self):
    return self.description

  def set_description(self, description):
    if isinstance(description, unicode):
      self.description = description.encode('utf-8')
    else:
      self.description = description

  def parse(self):
    parser = CaboCha.Parser()

    self.tree    = parser.parse(self.get_description())
    tokens = []
    for i in xrange(0, self.tree.size()):
      token = self.tree(i)
      surface = token.surface
      feature = token.feature
      mb_token = MeCabToken(surface,feature)
      tokens.append(mb_token)


  def get_raw_tree(self):
    print self.tree






if __name__ == '__main__':
  description = u'全ての単語は、いずれかの品詞に所属すると考えられる。だが、実際にどの品詞に含まれるかと問われれば、分類に悩むことも多い。そのため現在では認知心理学の意味研究からプロトタイプという考え方が導入されて説明されている。'
  cabocha = My_cabocha(description)
  cabocha.parse()
  print dir(cabocha.get_raw_tree())






