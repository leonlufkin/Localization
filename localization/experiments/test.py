def numberer():
  yield 1
  yield 2
  yield 3
  
def printer():
  for n in numberer():
    return n
    
  
if __name__ == '__main__':
  for n in numberer():
    print(n)
    
  # for n in printer():
  #   print(n)
    
  print( printer() )
  print( printer() )
  print( printer() )
  