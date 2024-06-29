import ftorch as torch
import time

if __name__ == "__main__":
    a = torch.tensor([[1,2,3],[1,2,3]])
    b = torch.tensor([[1,1,1],[1,2,3]])
    print(a+b)
    print(a)
    print(b)