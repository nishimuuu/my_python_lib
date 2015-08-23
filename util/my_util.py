#-*-coding: utf-8 -*-
import json,re,pprint
__author__ = '01010357'


# write code here...
class Util:
  def toJson(self,dict):
    return json.dumps(dict, indent = 2, sort_keys = True, ensure_ascii=False).replace('\\\\','')

  def pp(self,obj):
    pp = pprint.PrettyPrinter(indent=2, width=160)
    str = pp.pformat(obj)
    return re.sub(r'\\u([0-9a-f]{4})', lambda x: unichr(int('0x'+x.group(1),16)),str)




