import XCTest
@testable import SlopeCoach

final class RiderSelectionViewModelTests: XCTestCase {
    @MainActor
    func testIdleTapTransitionsThroughSelectingToSelected() async {
        let viewModel = makeViewModel(segmenter: SuccessfulSegmenter())
        let point = NormalizedVideoPoint(x: 0.75, y: 0.59)!

        let task = viewModel.selectRider(at: point)

        XCTAssertEqual(viewModel.selectionState, .selecting)
        XCTAssertFalse(viewModel.canContinue)
        await task?.value
        XCTAssertEqual(viewModel.selectionState, .selected)
        XCTAssertTrue(viewModel.canContinue)
        XCTAssertEqual(viewModel.formattedSelectionTime, "00:12")
    }

    @MainActor
    func testSelectingFailureTransitionsToFailed() async {
        let viewModel = makeViewModel(segmenter: FailingSegmenter())

        let task = viewModel.selectRider(at: NormalizedVideoPoint(x: 0.1, y: 0.1)!)

        XCTAssertEqual(viewModel.selectionState, .selecting)
        await task?.value
        XCTAssertEqual(viewModel.selectionState, .failed)
        XCTAssertFalse(viewModel.canContinue)
        XCTAssertNil(viewModel.currentMask)
    }

    @MainActor
    func testChangeReturnsSelectedStateToIdleWithoutChangingTime() async {
        let viewModel = makeViewModel(segmenter: SuccessfulSegmenter())
        let task = viewModel.selectRider(at: NormalizedVideoPoint(x: 0.75, y: 0.59)!)
        await task?.value
        viewModel.scrub(to: 21)

        viewModel.changeSelection()

        XCTAssertEqual(viewModel.selectionState, .idle)
        XCTAssertEqual(viewModel.currentTime, 21)
        XCTAssertNil(viewModel.selectedRider)
        XCTAssertNil(viewModel.currentMask)
        XCTAssertFalse(viewModel.canContinue)
    }

    @MainActor
    func testTimelineAndDisplayedTimeShareCurrentTime() {
        let viewModel = makeViewModel(segmenter: SuccessfulSegmenter())

        viewModel.beginScrubbing()
        viewModel.scrub(to: 27.8)

        XCTAssertEqual(viewModel.currentTime, 27.8, accuracy: 0.001)
        XCTAssertEqual(viewModel.formattedCurrentTime, "00:27")
        viewModel.endScrubbing()
    }

    @MainActor
    private func makeViewModel(segmenter: any RiderSegmenter) -> RiderSelectionViewModel {
        RiderSelectionViewModel(
            duration: 58,
            initialTime: 12,
            segmenter: segmenter
        )
    }
}

private struct SuccessfulSegmenter: RiderSegmenter {
    func selectRider(input: RiderSelectionInput) async throws -> RiderSelectionResult {
        let mask = RiderMask.normalizedPolygon([
            NormalizedVideoPoint(x: 0.7, y: 0.4)!,
            NormalizedVideoPoint(x: 0.8, y: 0.4)!,
            NormalizedVideoPoint(x: 0.8, y: 0.8)!,
            NormalizedVideoPoint(x: 0.7, y: 0.8)!
        ])
        let session = TestTrackingSession(
            selectionID: "test-selection",
            anchorTimestamp: input.timestamp,
            mask: mask
        )
        return RiderSelectionResult(
            selectionID: session.selectionID,
            timestamp: input.timestamp,
            initialMask: mask,
            confidence: nil,
            session: session
        )
    }
}

private struct FailingSegmenter: RiderSegmenter {
    func selectRider(input: RiderSelectionInput) async throws -> RiderSelectionResult {
        throw RiderSegmentationError.riderNotFound
    }
}

private struct TestTrackingSession: RiderTrackingSession {
    let selectionID: String
    let anchorTimestamp: TimeInterval
    let mask: RiderMask

    func mask(at timestamp: TimeInterval) async -> RiderMask? {
        abs(timestamp - anchorTimestamp) < 0.5 ? mask : nil
    }
}

