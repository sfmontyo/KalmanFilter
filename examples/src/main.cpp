#include <iostream>
#include <functional>

int testKfilter();
int testExample();
int kmatrixTest();
int kvectorTest();
int mfileTest();
void runTest( const char* testMethodName, std::function<int()> testMethod);

int main() {
    runTest("testKfilter", testKfilter);
    runTest("testExample", testExample);
    runTest("kmatrixTest", kmatrixTest);
    runTest("kvectorTest", kvectorTest);
    runTest("mfileTest", mfileTest);

    return 0;
}

void
runTest( const char* testMethodName, std::function<int()> testMethod) {
    std::cout << testMethodName << ": started " << std::endl;
    int exitCode = testMethod();
    std::cout << testMethodName << " finished. exit code: " << exitCode << std::endl;
}
