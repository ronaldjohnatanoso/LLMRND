# Dead Code Analysis Report
Generated: 2025-02-22

## Tools Used
- **knip**: Found unused exports and dependencies
- **depcheck**: Found unused dependencies
- **ts-prune**: Found unused TypeScript exports

## Findings

### SAFE - Can be deleted immediately

#### 1. Unused Type: `QueryRequest`
- **Location**: `src/types/index.ts:11:18`
- **Severity**: SAFE
- **Reason**: Only defined in types file, never imported anywhere
- **Impact**: None - unused interface

#### 2. Unused Dependency: `react-cytoscapejs`
- **Location**: `package.json:15:6`
- **Severity**: SAFE
- **Reason**: Project uses `cytoscape` directly, not the React wrapper
- **Impact**: Reduces bundle size

#### 3. Unused Function: `addNodes`
- **Location**: `src/lib/api.ts:43`
- **Severity**: SAFE
- **Reason**: Exported but never imported
- **Impact**: None

### CAUTION - Review before deleting

None found.

### DANGER - Do not delete

None found.

## Test Coverage
- No test files found in `src/` directory
- Cannot verify deletions with tests
- Will verify with build before/after each deletion

## Deletion Plan

1. ✅ Remove `QueryRequest` interface from types
2. ✅ Remove `react-cytoscapejs` from dependencies
3. ✅ Remove `addNodes` function from api.ts
4. ✅ Verify build after each change
