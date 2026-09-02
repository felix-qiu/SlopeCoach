import XCTest
@testable import SlopeCoach

final class VideoCoordinateMapperTests: XCTestCase {
    @MainActor
    func testMapsCenterPointToNormalizedVideoCenter() throws {
        let mapper = VideoCoordinateMapper(
            sourceSize: CGSize(width: 1920, height: 1080),
            containerSize: CGSize(width: 320, height: 200),
            contentMode: .aspectFit
        )

        let normalized = try XCTUnwrap(
            mapper.normalizedPoint(forDisplayPoint: CGPoint(x: 160, y: 100))
        )

        XCTAssertEqual(normalized.x, 0.5, accuracy: 0.0001)
        XCTAssertEqual(normalized.y, 0.5, accuracy: 0.0001)
    }

    @MainActor
    func testRejectsTapInsideLetterboxOutsideVideoContent() {
        let mapper = VideoCoordinateMapper(
            sourceSize: CGSize(width: 1920, height: 1080),
            containerSize: CGSize(width: 200, height: 300),
            contentMode: .aspectFit
        )

        XCTAssertNil(mapper.normalizedPoint(forDisplayPoint: CGPoint(x: 100, y: 20)))
    }

    @MainActor
    func testAspectRatioMismatchMapsVideoEdgesCorrectly() throws {
        let mapper = VideoCoordinateMapper(
            sourceSize: CGSize(width: 1920, height: 1080),
            containerSize: CGSize(width: 300, height: 300),
            contentMode: .aspectFit
        )
        let videoRect = mapper.displayedVideoRect

        let topLeft = try XCTUnwrap(
            mapper.normalizedPoint(forDisplayPoint: CGPoint(x: videoRect.minX, y: videoRect.minY))
        )
        let bottomRight = try XCTUnwrap(
            mapper.normalizedPoint(forDisplayPoint: CGPoint(x: videoRect.maxX, y: videoRect.maxY))
        )

        XCTAssertEqual(topLeft.x, 0, accuracy: 0.0001)
        XCTAssertEqual(topLeft.y, 0, accuracy: 0.0001)
        XCTAssertEqual(bottomRight.x, 1, accuracy: 0.0001)
        XCTAssertEqual(bottomRight.y, 1, accuracy: 0.0001)
    }

    @MainActor
    func testMaskAndTapUseSameCoordinateMapping() throws {
        let mapper = VideoCoordinateMapper(
            sourceSize: CGSize(width: 1400, height: 933),
            containerSize: CGSize(width: 361, height: 225.625),
            contentMode: .aspectFit
        )
        let normalized = NormalizedVideoPoint(x: 0.75, y: 0.59)!
        let displayPoint = mapper.displayPoint(for: normalized)
        let roundTrip = try XCTUnwrap(mapper.normalizedPoint(forDisplayPoint: displayPoint))
        let mask = RiderMask.normalizedPolygon([normalized])

        XCTAssertEqual(roundTrip.x, normalized.x, accuracy: 0.0001)
        XCTAssertEqual(roundTrip.y, normalized.y, accuracy: 0.0001)
        XCTAssertEqual(mapper.displayPoints(for: mask).first, displayPoint)
    }
}

