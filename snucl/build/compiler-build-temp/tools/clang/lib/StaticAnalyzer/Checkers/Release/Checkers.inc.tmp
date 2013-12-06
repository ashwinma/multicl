
#ifdef GET_CHECKERS
CHECKER("core.AdjustRet", AdjustedReturnValueChecker, AdjustedReturnValueChecker.cpp, "Check to see if the return value of a function call is different than the caller expects", false)
CHECKER("debug.Stats", AnalyzerStatsChecker, AnalyzerStatsChecker.cpp, "Emit warnings with analyzer statistics", false)
CHECKER("core.experimental.ArrayBound", ArrayBoundChecker, ArrayBoundChecker.cpp, "Check for an out-of-bound pointer being returned to callers", true)
CHECKER("core.experimental.Overflow", ArrayBoundCheckerV2, ArrayBoundCheckerV2.cpp, "Warn about buffer overflows", true)
CHECKER("core.AttrNonNull", AttrNonNullChecker, AttrNonNullChecker.cpp, "Check for arguments declared to have nonnull attribute", false)
CHECKER("core.BuiltinFunc", BuiltinFunctionChecker, BuiltinFunctionChecker.cpp, "Evaluate clang builtin functions", false)
CHECKER("macosx.CFError", CFErrorChecker, NSErrorChecker.cpp, "Check usage of CFErrorRef* parameters", false)
CHECKER("debug.DumpCFG", CFGDumper, DebugCheckers.cpp, "Display Control-Flow Graphs", false)
CHECKER("debug.ViewCFG", CFGViewer, DebugCheckers.cpp, "View Control-Flow Graphs using GraphViz", false)
CHECKER("macosx.CFNumber", CFNumberCreateChecker, BasicObjCFoundationChecks.cpp, "Check for proper uses of CFNumberCreate", false)
CHECKER("macosx.CFRetainRelease", CFRetainReleaseChecker, BasicObjCFoundationChecks.cpp, "Check for null arguments to CFRetain/CFRelease", false)
CHECKER("core.experimental.CString", CStringChecker, CStringChecker.cpp, "Check calls to functions in <string.h>", true)
CHECKER("core.CallAndMsg", CallAndMessageChecker, CallAndMessageChecker.cpp, "Check for errors of call and objc message expressions", false)
CHECKER("core.experimental.CastSize", CastSizeChecker, CastSizeChecker.cpp, "Check when casting a malloc'ed type T, whether the size is a multiple of the size of T", true)
CHECKER("core.experimental.CastToStruct", CastToStructChecker, CastToStructChecker.cpp, "Check for cast from non-struct pointer to struct pointer", true)
CHECKER("unix.experimental.Chroot", ChrootChecker, ChrootChecker.cpp, "Check improper use of chroot", true)
CHECKER("cocoa.ClassRelease", ClassReleaseChecker, BasicObjCFoundationChecks.cpp, "Check for sending 'retain', 'release', or 'autorelease' directly to a Class", false)
CHECKER("DeadStores", DeadStoresChecker, DeadStoresChecker.cpp, "Check for values stored to a variables that are never read afterwards", false)
CHECKER("core.Deref", DereferenceChecker, DereferenceChecker.cpp, "Check for null pointers at loads and stores", false)
CHECKER("core.DivZero", DivZeroChecker, DivZeroChecker.cpp, "Check for division by zeros", false)
CHECKER("core.experimental.FixedAddr", FixedAddressChecker, FixedAddressChecker.cpp, "Check for assignment of a fixed address to a pointer", true)
CHECKER("core.experimental.IdempotentOps", IdempotentOperationChecker, IdempotentOperationChecker.cpp, "Warn about idempotent operations", true)
CHECKER("llvm.Conventions", LLVMConventionsChecker, LLVMConventionsChecker.cpp, "Check code for LLVM codebase conventions", false)
CHECKER("debug.DumpLiveVars", LiveVariablesDumper, DebugCheckers.cpp, "Print results of live variable analysis", false)
CHECKER("macosx.API", MacOSXAPIChecker, MacOSXAPIChecker.cpp, "Check for proper uses of various Mac OS X APIs", false)
CHECKER("core.experimental.Malloc", MallocChecker, MallocChecker.cpp, "Check for potential memory leaks, double free, and use-after-free problems", true)
CHECKER("cocoa.NSAutoreleasePool", NSAutoreleasePoolChecker, NSAutoreleasePoolChecker.cpp, "Warn for subpar uses of NSAutoreleasePool", false)
CHECKER("cocoa.NSError", NSErrorChecker, NSErrorChecker.cpp, "Check usage of NSError** parameters", false)
CHECKER("cocoa.NilArg", NilArgChecker, BasicObjCFoundationChecks.cpp, "Check for prohibited nil arguments to ObjC method calls", false)
CHECKER("core.NoReturnFunc", NoReturnFunctionChecker, NoReturnFunctionChecker.cpp, "Evaluate functions that do not return to the caller", false)
CHECKER("macosx.AtomicCAS", OSAtomicChecker, OSAtomicChecker.cpp, "Evaluate calls to OSAtomic functions", false)
CHECKER("cocoa.AtSync", ObjCAtSyncChecker, ObjCAtSyncChecker.cpp, "Check for null pointers used as mutexes for @synchronized", false)
CHECKER("cocoa.experimental.Dealloc", ObjCDeallocChecker, CheckObjCDealloc.cpp, "Warn about Objective-C classes that lack a correct implementation of -dealloc", true)
CHECKER("cocoa.MethodSigs", ObjCMethSigsChecker, CheckObjCInstMethSignature.cpp, "Warn about Objective-C method signatures with type incompatibilities", false)
CHECKER("cocoa.SelfInit", ObjCSelfInitChecker, ObjCSelfInitChecker.cpp, "Check that 'self' is propely initialized inside an initializer method", false)
CHECKER("cocoa.UnusedIvars", ObjCUnusedIvarsChecker, ObjCUnusedIVarsChecker.cpp, "Warn about private ivars that are never used", false)
CHECKER("core.experimental.PointerArithm", PointerArithChecker, PointerArithChecker, "Check for pointer arithmetic on locations other than array elements", true)
CHECKER("core.experimental.PointerSub", PointerSubChecker, PointerSubChecker, "Check for pointer subtractions on two pointers pointing to different memory chunks", true)
CHECKER("unix.experimental.PthreadLock", PthreadLockChecker, PthreadLockChecker.cpp, "Simple lock -> unlock checker", true)
CHECKER("core.experimental.ReturnPtrRange", ReturnPointerRangeChecker, ReturnPointerRangeChecker.cpp, "Check for an out-of-bound pointer being returned to callers", true)
CHECKER("core.UndefReturn", ReturnUndefChecker, ReturnUndefChecker.cpp, "Check for undefined or garbage values being returned to the caller", false)
CHECKER("core.experimental.SecuritySyntactic", SecuritySyntaxChecker, CheckSecuritySyntaxOnly.cpp, "Perform quick security checks that require no data flow", true)
CHECKER("core.experimental.SizeofPtr", SizeofPointerChecker, CheckSizeofPointer.cpp, "Warn about unintended use of sizeof() on pointer expressions", true)
CHECKER("core.StackAddrEscape", StackAddrEscapeChecker, StackAddrEscapeChecker.cpp, "Check that addresses to stack memory do not escape the function", false)
CHECKER("unix.experimental.Stream", StreamChecker, StreamChecker.cpp, "Check stream handling functions", true)
CHECKER("core.UndefBranch", UndefBranchChecker, UndefBranchChecker.cpp, "Check for undefined branch conditions", false)
CHECKER("core.UndefBlockVar", UndefCapturedBlockVarChecker, UndefCapturedBlockVarChecker.cpp, "Check for blocks that capture uninitialized values", false)
CHECKER("core.UndefBinOpResult", UndefResultChecker, UndefResultChecker.cpp, "Check for undefined results of non-assignment binary operators", false)
CHECKER("core.UndefArraySubscript", UndefinedArraySubscriptChecker, UndefinedArraySubscriptChecker.cpp, "Check for undefined array subscripts", false)
CHECKER("core.UndefAssign", UndefinedAssignmentChecker, UndefinedAssignmentChecker.cpp, "Check for assigning undefined values", false)
CHECKER("unix.API", UnixAPIChecker, UnixAPIChecker.cpp, "Check calls to various UNIX/Posix functions", false)
CHECKER("core.experimental.UnreachableCode", UnreachableCodeChecker, UnreachableCodeChecker.cpp, "Check unreachable code", true)
CHECKER("core.VLASize", VLASizeChecker, VLASizeChecker.cpp, "Check for declarations of VLA of undefined or zero size", false)
#endif // GET_CHECKERS


#ifdef GET_MEMBER_ARRAYS
static const short CheckerArray0[] = { 17, -1 };
static const short CheckerArray1[] = { 2, 3, 11, 13, 14, 15, 20, 21, 25, 32, 36, 37, 38, 39, 41, 42, 44, 51, -1 };
static const short SubPackageArray2[] = { 4, 7, 5, 10, 8, 3, 11, 6, 9, -1 };
static const short CheckerArray3[] = { 31, -1 };
static const short CheckerArray4[] = { 16, -1 };
static const short CheckerArray5[] = { 33, -1 };
static const short CheckerArray6[] = { 26, -1 };
static const short CheckerArray7[] = { 27, -1 };
static const short CheckerArray8[] = { 28, -1 };
static const short CheckerArray9[] = { 34, -1 };
static const short CheckerArray10[] = { 35, -1 };
static const short SubPackageArray11[] = { 12, -1 };
static const short CheckerArray12[] = { 32, -1 };
static const short SubPackageArray13[] = { 23, 19, 29, 22, 27, 17, 14, 15, 28, 18, 24, 20, 26, 25, 21, 16, -1 };
static const short CheckerArray14[] = { 0, -1 };
static const short CheckerArray15[] = { 4, -1 };
static const short CheckerArray16[] = { 5, -1 };
static const short CheckerArray17[] = { 12, -1 };
static const short CheckerArray18[] = { 18, -1 };
static const short CheckerArray19[] = { 19, -1 };
static const short CheckerArray20[] = { 29, -1 };
static const short CheckerArray21[] = { 43, -1 };
static const short CheckerArray22[] = { 48, -1 };
static const short CheckerArray23[] = { 49, -1 };
static const short CheckerArray24[] = { 47, -1 };
static const short CheckerArray25[] = { 46, -1 };
static const short CheckerArray26[] = { 45, -1 };
static const short CheckerArray27[] = { 40, -1 };
static const short CheckerArray28[] = { 52, -1 };
static const short SubPackageArray29[] = { 40, 30, 42, 41, 37, 31, 43, 34, 38, 39, 32, 36, 33, 35, -1 };
static const short CheckerArray30[] = { 2, -1 };
static const short CheckerArray31[] = { 11, -1 };
static const short CheckerArray32[] = { 13, -1 };
static const short CheckerArray33[] = { 14, -1 };
static const short CheckerArray34[] = { 20, -1 };
static const short CheckerArray35[] = { 21, -1 };
static const short CheckerArray36[] = { 25, -1 };
static const short CheckerArray37[] = { 3, -1 };
static const short CheckerArray38[] = { 36, -1 };
static const short CheckerArray39[] = { 37, -1 };
static const short CheckerArray40[] = { 39, -1 };
static const short CheckerArray41[] = { 41, -1 };
static const short CheckerArray42[] = { 42, -1 };
static const short CheckerArray43[] = { 51, -1 };
static const short SubPackageArray44[] = { 47, 45, 48, 46, -1 };
static const short CheckerArray45[] = { 7, -1 };
static const short CheckerArray46[] = { 23, -1 };
static const short CheckerArray47[] = { 1, -1 };
static const short CheckerArray48[] = { 8, -1 };
static const short SubPackageArray49[] = { 50, -1 };
static const short CheckerArray50[] = { 22, -1 };
static const short SubPackageArray51[] = { 54, 53, 56, 55, 52, -1 };
static const short CheckerArray52[] = { 24, -1 };
static const short CheckerArray53[] = { 30, -1 };
static const short CheckerArray54[] = { 6, -1 };
static const short CheckerArray55[] = { 9, -1 };
static const short CheckerArray56[] = { 10, -1 };
static const short SubPackageArray57[] = { 58, 59, -1 };
static const short CheckerArray58[] = { 50, -1 };
static const short SubPackageArray59[] = { 61, 62, 60, -1 };
static const short CheckerArray60[] = { 15, -1 };
static const short CheckerArray61[] = { 38, -1 };
static const short CheckerArray62[] = { 44, -1 };
#endif // GET_MEMBER_ARRAYS


#ifdef GET_CHECKNAME_TABLE
  { "DeadStores",                          CheckerArray0, 0, false },
  { "all-experimental",                    CheckerArray1, 0, false },
  { "cocoa",                               0, SubPackageArray2, false },
  { "cocoa.AtSync",                        CheckerArray3, 0, false },
  { "cocoa.ClassRelease",                  CheckerArray4, 0, false },
  { "cocoa.MethodSigs",                    CheckerArray5, 0, false },
  { "cocoa.NSAutoreleasePool",             CheckerArray6, 0, false },
  { "cocoa.NSError",                       CheckerArray7, 0, false },
  { "cocoa.NilArg",                        CheckerArray8, 0, false },
  { "cocoa.SelfInit",                      CheckerArray9, 0, false },
  { "cocoa.UnusedIvars",                   CheckerArray10, 0, false },
  { "cocoa.experimental",                  0, SubPackageArray11, true },
  { "cocoa.experimental.Dealloc",          CheckerArray12, 0, false },
  { "core",                                0, SubPackageArray13, false },
  { "core.AdjustRet",                      CheckerArray14, 0, false },
  { "core.AttrNonNull",                    CheckerArray15, 0, false },
  { "core.BuiltinFunc",                    CheckerArray16, 0, false },
  { "core.CallAndMsg",                     CheckerArray17, 0, false },
  { "core.Deref",                          CheckerArray18, 0, false },
  { "core.DivZero",                        CheckerArray19, 0, false },
  { "core.NoReturnFunc",                   CheckerArray20, 0, false },
  { "core.StackAddrEscape",                CheckerArray21, 0, false },
  { "core.UndefArraySubscript",            CheckerArray22, 0, false },
  { "core.UndefAssign",                    CheckerArray23, 0, false },
  { "core.UndefBinOpResult",               CheckerArray24, 0, false },
  { "core.UndefBlockVar",                  CheckerArray25, 0, false },
  { "core.UndefBranch",                    CheckerArray26, 0, false },
  { "core.UndefReturn",                    CheckerArray27, 0, false },
  { "core.VLASize",                        CheckerArray28, 0, false },
  { "core.experimental",                   0, SubPackageArray29, true },
  { "core.experimental.ArrayBound",        CheckerArray30, 0, false },
  { "core.experimental.CString",           CheckerArray31, 0, false },
  { "core.experimental.CastSize",          CheckerArray32, 0, false },
  { "core.experimental.CastToStruct",      CheckerArray33, 0, false },
  { "core.experimental.FixedAddr",         CheckerArray34, 0, false },
  { "core.experimental.IdempotentOps",     CheckerArray35, 0, false },
  { "core.experimental.Malloc",            CheckerArray36, 0, false },
  { "core.experimental.Overflow",          CheckerArray37, 0, true },
  { "core.experimental.PointerArithm",     CheckerArray38, 0, false },
  { "core.experimental.PointerSub",        CheckerArray39, 0, false },
  { "core.experimental.ReturnPtrRange",    CheckerArray40, 0, false },
  { "core.experimental.SecuritySyntactic", CheckerArray41, 0, false },
  { "core.experimental.SizeofPtr",         CheckerArray42, 0, false },
  { "core.experimental.UnreachableCode",   CheckerArray43, 0, true },
  { "debug",                               0, SubPackageArray44, false },
  { "debug.DumpCFG",                       CheckerArray45, 0, false },
  { "debug.DumpLiveVars",                  CheckerArray46, 0, false },
  { "debug.Stats",                         CheckerArray47, 0, false },
  { "debug.ViewCFG",                       CheckerArray48, 0, false },
  { "llvm",                                0, SubPackageArray49, false },
  { "llvm.Conventions",                    CheckerArray50, 0, false },
  { "macosx",                              0, SubPackageArray51, false },
  { "macosx.API",                          CheckerArray52, 0, false },
  { "macosx.AtomicCAS",                    CheckerArray53, 0, false },
  { "macosx.CFError",                      CheckerArray54, 0, false },
  { "macosx.CFNumber",                     CheckerArray55, 0, false },
  { "macosx.CFRetainRelease",              CheckerArray56, 0, false },
  { "unix",                                0, SubPackageArray57, false },
  { "unix.API",                            CheckerArray58, 0, false },
  { "unix.experimental",                   0, SubPackageArray59, true },
  { "unix.experimental.Chroot",            CheckerArray60, 0, false },
  { "unix.experimental.PthreadLock",       CheckerArray61, 0, false },
  { "unix.experimental.Stream",            CheckerArray62, 0, false },
#endif // GET_CHECKNAME_TABLE

